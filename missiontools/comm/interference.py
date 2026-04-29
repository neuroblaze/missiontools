"""Interference analysis between space networks."""

from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt

from .antenna import AbstractAntenna
from ..condition.condition import AbstractCondition
from ..orbit.access import (
    earth_access_intervals,
    space_to_space_access,
    space_to_space_access_intervals,
)
from ..orbit.constants import EARTH_MEAN_RADIUS
from ..orbit.frames import geodetic_to_ecef, ecef_to_eci
from ..orbit.propagation import propagate_analytical

_C = 299_792_458.0


def _host_eci(
    host,
    t_arr: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    from ..spacecraft import Spacecraft
    from ..ground_station import GroundStation

    if isinstance(host, Spacecraft):
        r, v = propagate_analytical(
            t_arr, **host.keplerian_params, propagator_type=host.propagator_type
        )
        return r, v
    elif isinstance(host, GroundStation):
        r_ecef = geodetic_to_ecef(np.radians(host.lat), np.radians(host.lon), host.alt)
        n = len(t_arr)
        r_ecef_tiled = np.tile(r_ecef, (n, 1))
        r_eci = ecef_to_eci(r_ecef_tiled, t_arr)
        v_eci = np.zeros((n, 3))
        return r_eci, v_eci
    else:
        raise TypeError(f"Unsupported host type: {type(host).__name__!r}")


def _get_host_type(host):
    from ..spacecraft import Spacecraft
    from ..ground_station import GroundStation

    if isinstance(host, Spacecraft):
        return "sc"
    elif isinstance(host, GroundStation):
        return "gs"
    raise TypeError(f"Unsupported host type: {type(host).__name__!r}")


def _gs_gs_access_intervals(
    gs_a,
    gs_b,
    t_start: np.datetime64,
    t_end: np.datetime64,
) -> list[tuple[np.datetime64, np.datetime64]]:
    r_a_ecef = geodetic_to_ecef(np.radians(gs_a.lat), np.radians(gs_a.lon), gs_a.alt)
    r_b_ecef = geodetic_to_ecef(np.radians(gs_b.lat), np.radians(gs_b.lon), gs_b.alt)
    clear = bool(
        space_to_space_access(
            r_a_ecef.reshape(1, 3),
            r_b_ecef.reshape(1, 3),
            EARTH_MEAN_RADIUS,
        )[0]
    )
    if clear:
        return [(t_start, t_end)]
    return []


def _get_access_intervals(
    cache: dict,
    host_a,
    host_b,
    t_start: np.datetime64,
    t_end: np.datetime64,
    max_step: np.timedelta64,
) -> list[tuple[np.datetime64, np.datetime64]]:
    key = (id(host_a), id(host_b))
    if key in cache:
        return cache[key]

    type_a = _get_host_type(host_a)
    type_b = _get_host_type(host_b)

    if type_a == "sc" and type_b == "sc":
        intervals = space_to_space_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params_1=host_a.keplerian_params,
            keplerian_params_2=host_b.keplerian_params,
            body_radius=host_a.central_body_radius,
            propagator_type_1=host_a.propagator_type,
            propagator_type_2=host_b.propagator_type,
            max_step=max_step,
        )
    elif type_a == "sc" and type_b == "gs":
        intervals = earth_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params=host_a.keplerian_params,
            lat=np.radians(host_b.lat),
            lon=np.radians(host_b.lon),
            alt=host_b.alt,
            el_min=0.0,
            propagator_type=host_a.propagator_type,
            max_step=max_step,
        )
    elif type_a == "gs" and type_b == "sc":
        intervals = earth_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params=host_b.keplerian_params,
            lat=np.radians(host_a.lat),
            lon=np.radians(host_a.lon),
            alt=host_a.alt,
            el_min=0.0,
            propagator_type=host_b.propagator_type,
            max_step=max_step,
        )
    elif type_a == "gs" and type_b == "gs":
        intervals = _gs_gs_access_intervals(host_a, host_b, t_start, t_end)
    else:
        intervals = []

    cache[key] = intervals
    return intervals


def _intersect_intervals(
    intervals_a: list[tuple[np.datetime64, np.datetime64]],
    intervals_b: list[tuple[np.datetime64, np.datetime64]],
) -> list[tuple[np.datetime64, np.datetime64]]:
    result = []
    i, j = 0, 0
    while i < len(intervals_a) and j < len(intervals_b):
        start = max(intervals_a[i][0], intervals_b[j][0])
        end = min(intervals_a[i][1], intervals_b[j][1])
        if start < end:
            result.append((start, end))
        if intervals_a[i][1] < intervals_b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _find_exceedance_runs(
    psd: npt.NDArray[np.floating],
    threshold: float,
) -> list[tuple[int, int]]:
    above = psd >= threshold
    if not np.any(above):
        return []

    runs = []
    in_run = False
    start = 0
    for k in range(len(above)):
        if above[k] and not in_run:
            start = k
            in_run = True
        elif not above[k] and in_run:
            runs.append((start, k - 1))
            in_run = False
    if in_run:
        runs.append((start, len(above) - 1))
    return runs


class InterferenceAnalysis:
    """Analyze interference risk between space networks.

    Parameters
    ----------
    f_MHz : float
        Centre frequency of the channel in MHz.
    """

    def __init__(self, f_MHz: float) -> None:
        if f_MHz <= 0:
            raise ValueError(f"f_MHz must be positive, got {f_MHz}")
        self._f_Hz = float(f_MHz) * 1e6
        self._victim_txs: list[dict] = []
        self._victim_rxs: list[dict] = []
        self._interfering_txs: list[dict] = []

    def add_victim_tx(
        self,
        name: str,
        antenna: AbstractAntenna,
        tx_psd: float,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._victim_txs.append(
            {
                "name": name,
                "antenna": antenna,
                "tx_psd": float(tx_psd),
                "condition": condition,
            }
        )

    def add_victim_rx(
        self,
        name: str,
        antenna: AbstractAntenna,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._victim_rxs.append(
            {
                "name": name,
                "antenna": antenna,
                "condition": condition,
            }
        )

    def add_interfering_tx(
        self,
        name: str,
        antenna: AbstractAntenna,
        tx_psd: float,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._interfering_txs.append(
            {
                "name": name,
                "antenna": antenna,
                "tx_psd": float(tx_psd),
                "condition": condition,
            }
        )

    def compute(
        self,
        psd_threshold: float,
        start_time: npt.ArrayLike,
        end_time: npt.ArrayLike,
        max_step: float = 10.0,
        event_step: float = 1.0,
    ) -> list[dict]:
        """Compute interference events.

        Parameters
        ----------
        psd_threshold : float
            Interference PSD threshold in dBW/Hz.
        start_time : array_like of datetime64
            Start of the analysis window.
        end_time : array_like of datetime64
            End of the analysis window.
        max_step : float, optional
            Coarse scan step for access interval detection in seconds.
            Default 10.0.
        event_step : float, optional
            Sampling step within candidate intervals in seconds.
            Default 1.0.

        Returns
        -------
        list[dict]
            Each dict represents an interference event with keys:
            ``victim_tx``, ``victim_rx``, ``interfering_tx`` (str),
            ``start_time``, ``end_time`` (datetime64),
            ``max_interferer_psd`` (float, dBW/Hz),
            ``times`` (ndarray of datetime64),
            ``interferer_psd``, ``victim_psd`` (ndarray, dBW/Hz).
        """
        if not self._victim_txs:
            raise ValueError("No victim transmitters have been added.")
        if not self._victim_rxs:
            raise ValueError("No victim receivers have been added.")
        if not self._interfering_txs:
            raise ValueError("No interfering transmitters have been added.")

        t_start = np.asarray(start_time, dtype="datetime64[us]")
        t_end = np.asarray(end_time, dtype="datetime64[us]")
        max_step_td = np.timedelta64(int(round(max_step * 1e6)), "us")
        event_step_td = np.timedelta64(int(round(event_step * 1e6)), "us")

        access_cache: dict = {}
        events: list[dict] = []

        for vtx, vrx, itx in product(
            self._victim_txs, self._victim_rxs, self._interfering_txs
        ):
            vtx_host = vtx["antenna"].host
            vrx_host = vrx["antenna"].host
            itx_host = itx["antenna"].host

            victim_intervals = _get_access_intervals(
                access_cache,
                vtx_host,
                vrx_host,
                t_start,
                t_end,
                max_step_td,
            )
            interferer_intervals = _get_access_intervals(
                access_cache,
                itx_host,
                vrx_host,
                t_start,
                t_end,
                max_step_td,
            )

            candidate_intervals = _intersect_intervals(
                victim_intervals,
                interferer_intervals,
            )

            for ci_start, ci_end in candidate_intervals:
                duration_us = int((ci_end - ci_start) / np.timedelta64(1, "us"))
                step_us = int(event_step_td / np.timedelta64(1, "us"))
                if step_us <= 0:
                    step_us = 1

                n_samples = duration_us // step_us + 1
                offsets_us = np.arange(n_samples, dtype=np.int64) * step_us
                sample_times = ci_start + offsets_us.astype("timedelta64[us]")

                if offsets_us[-1] < duration_us:
                    sample_times = np.append(
                        sample_times,
                        ci_start + np.timedelta64(duration_us, "us"),
                    )

                mask = np.ones(len(sample_times), dtype=bool)
                if vtx["condition"] is not None:
                    mask &= vtx["condition"].at(sample_times)
                if vrx["condition"] is not None:
                    mask &= vrx["condition"].at(sample_times)
                if itx["condition"] is not None:
                    mask &= itx["condition"].at(sample_times)

                if not np.any(mask):
                    continue

                times_masked = sample_times[mask]

                r_vtx, v_vtx = _host_eci(vtx_host, times_masked)
                r_vrx, v_vrx = _host_eci(vrx_host, times_masked)
                r_itx, v_itx = _host_eci(itx_host, times_masked)

                delta_v_vrx = r_vrx - r_vtx
                range_v = np.linalg.norm(delta_v_vrx, axis=1)
                fspl_v = 20.0 * np.log10(4.0 * np.pi * range_v * self._f_Hz / _C)

                delta_i_vrx = r_vrx - r_itx
                range_i = np.linalg.norm(delta_i_vrx, axis=1)
                fspl_i = 20.0 * np.log10(4.0 * np.pi * range_i * self._f_Hz / _C)

                g_vtx = vtx["antenna"].gain(
                    times_masked,
                    delta_v_vrx,
                    frame="eci",
                    r_eci=r_vtx,
                    v_eci=v_vtx,
                )
                g_vrx_v = vrx["antenna"].gain(
                    times_masked,
                    -delta_v_vrx,
                    frame="eci",
                    r_eci=r_vrx,
                    v_eci=v_vrx,
                )

                g_itx = itx["antenna"].gain(
                    times_masked,
                    delta_i_vrx,
                    frame="eci",
                    r_eci=r_itx,
                    v_eci=v_itx,
                )
                g_vrx_i = vrx["antenna"].gain(
                    times_masked,
                    -delta_i_vrx,
                    frame="eci",
                    r_eci=r_vrx,
                    v_eci=v_vrx,
                )

                victim_psd = vtx["tx_psd"] + g_vtx - fspl_v + g_vrx_v
                interf_psd = itx["tx_psd"] + g_itx - fspl_i + g_vrx_i

                runs = _find_exceedance_runs(interf_psd, psd_threshold)

                for run_start, run_end in runs:
                    events.append(
                        {
                            "victim_tx": vtx["name"],
                            "victim_rx": vrx["name"],
                            "interfering_tx": itx["name"],
                            "start_time": times_masked[run_start],
                            "end_time": times_masked[run_end],
                            "max_interferer_psd": float(
                                np.max(interf_psd[run_start : run_end + 1])
                            ),
                            "times": times_masked[run_start : run_end + 1].copy(),
                            "interferer_psd": interf_psd[
                                run_start : run_end + 1
                            ].copy(),
                            "victim_psd": victim_psd[run_start : run_end + 1].copy(),
                        }
                    )

        return events
