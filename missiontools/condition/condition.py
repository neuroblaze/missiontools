"""
Conditions
==========
Boolean time-domain predicates with built-in caching.

A :class:`AbstractCondition` is a callable with one method,
:meth:`~AbstractCondition.at`, that returns a boolean array indicating
whether the condition holds at each requested time.  Conditions capture
any required external state (spacecraft, ground stations, ...) at
construction time, so the public API depends only on time.

Hierarchy
---------
:class:`AbstractCondition` (ABC)
└── :class:`SpaceGroundAccessCondition`
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import numpy.typing as npt

from ..cache import cached_propagate_analytical
from ..orbit.access import earth_access


class AbstractCondition(ABC):
    """Base class for boolean time-domain conditions.

    Subclasses implement :meth:`_compute` and :meth:`__repr__`.  The base
    class handles input coercion, scalar/array shape tracking, and a
    small per-instance count-based LRU cache keyed on the SHA-256 digest
    of the requested time array.

    Parameters
    ----------
    cache_size : int, optional
        Maximum number of distinct time arrays whose results are cached.
        Default 16.  Set to 0 to disable caching.

    Notes
    -----
    Subclasses can bypass caching entirely by overriding :meth:`at` rather
    than :meth:`_compute`.
    """

    def __init__(self, cache_size: int = 16) -> None:
        if cache_size < 0:
            raise ValueError(
                f"cache_size must be non-negative, got {cache_size}"
            )
        self._cache_size = cache_size
        self._cache: OrderedDict[bytes, npt.NDArray[np.bool_]] = OrderedDict()

    @abstractmethod
    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        """Evaluate the condition at the given times.

        Parameters
        ----------
        t : ndarray of datetime64[us], shape (N,)
            Times at which to evaluate the condition.

        Returns
        -------
        ndarray of bool, shape (N,)
        """

    @abstractmethod
    def __repr__(self) -> str: ...

    def at(self, t: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Evaluate the condition at one or more times.

        Parameters
        ----------
        t : array_like of datetime64, shape (N,) or scalar
            Time(s) at which to evaluate.

        Returns
        -------
        ndarray of bool, shape (N,) or scalar bool
            True where the condition holds.
        """
        t_in    = np.asarray(t, dtype='datetime64[us]')
        scalar  = t_in.ndim == 0
        t_arr   = np.atleast_1d(t_in)

        if self._cache_size > 0:
            key = hashlib.sha256(t_arr.tobytes()).digest()
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return bool(cached[0]) if scalar else cached
            result = np.asarray(self._compute(t_arr), dtype=bool)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            result = np.asarray(self._compute(t_arr), dtype=bool)

        return bool(result[0]) if scalar else result


class SpaceGroundAccessCondition(AbstractCondition):
    """True when a spacecraft is visible from a ground station.

    Visibility is the standard above-horizon test: the elevation angle
    from the geodetic up-direction at the ground station to the
    spacecraft must meet or exceed ``el_min``.  Earth blockage is implicit
    for ``el_min >= 0``.

    Parameters
    ----------
    spacecraft : Spacecraft
        The spacecraft whose visibility is being tested.
    ground_station : GroundStation
        The observing ground station.
    el_min : float, optional
        Minimum elevation angle (degrees).  Default 5.0.

    Raises
    ------
    TypeError
        If ``spacecraft`` is not a :class:`~missiontools.Spacecraft` or
        ``ground_station`` is not a :class:`~missiontools.GroundStation`.

    Examples
    --------
    ::

        from missiontools import Spacecraft, GroundStation
        from missiontools.condition import SpaceGroundAccessCondition

        sc = Spacecraft(...)
        gs = GroundStation(lat=51.5, lon=-0.1)
        cond = SpaceGroundAccessCondition(sc, gs, el_min=5.0)
        cond.at(np.datetime64('2025-01-01', 'us'))   # -> bool
    """

    def __init__(self, spacecraft, ground_station, el_min: float = 5.0) -> None:
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation
        if not isinstance(spacecraft, Spacecraft):
            raise TypeError(
                f"spacecraft must be a Spacecraft instance, "
                f"got {type(spacecraft).__name__!r}"
            )
        if not isinstance(ground_station, GroundStation):
            raise TypeError(
                f"ground_station must be a GroundStation instance, "
                f"got {type(ground_station).__name__!r}"
            )
        if not np.isfinite(el_min):
            raise ValueError(f"el_min must be finite, got {el_min}")
        super().__init__()
        self._sc = spacecraft
        self._gs = ground_station
        self._el_min_deg = float(el_min)
        self._el_min_rad = np.radians(self._el_min_deg)

    def __repr__(self) -> str:
        return (
            f"SpaceGroundAccessCondition("
            f"spacecraft={self._sc!r}, ground_station={self._gs!r}, "
            f"el_min={self._el_min_deg})"
        )

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        r, _ = cached_propagate_analytical(
            t,
            **self._sc.keplerian_params,
            propagator_type=self._sc.propagator_type,
        )
        return earth_access(
            r,
            lat    = np.radians(self._gs.lat),
            lon    = np.radians(self._gs.lon),
            alt    = self._gs.alt,
            el_min = self._el_min_rad,
            frame  = 'eci',
            t      = t,
        )
