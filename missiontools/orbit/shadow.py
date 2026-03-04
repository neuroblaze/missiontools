"""Cylindrical shadow model for eclipse detection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .constants import EARTH_SEMI_MAJOR_AXIS
from .frames import sun_vec_eci


def in_sunlight(
    r_eci: npt.ArrayLike,
    t: np.datetime64 | npt.NDArray[np.datetime64],
    body_radius: float = EARTH_SEMI_MAJOR_AXIS,
) -> np.bool_ | npt.NDArray[np.bool_]:
    """Determine whether a spacecraft is in sunlight using a cylindrical shadow model.

    Parameters
    ----------
    r_eci : array_like, shape (3,) or (N, 3)
        Spacecraft position(s) in the ECI frame (m).
    t : datetime64 or array of datetime64
        Epoch(s) for computing the Sun direction.
    body_radius : float, optional
        Central body equatorial radius (m).  Defaults to Earth WGS84.

    Returns
    -------
    np.bool_ or ndarray of bool
        ``True`` where the spacecraft is in sunlight.  Scalar input gives a
        scalar result; array input gives an ``(N,)`` array.
    """
    r = np.asarray(r_eci, dtype=np.float64)
    scalar = r.ndim == 1
    r_2d = np.atleast_2d(r)                          # (N, 3)

    s = sun_vec_eci(t)                                # (3,) or (N, 3)
    s_2d = np.atleast_2d(s)                           # (N, 3)

    # Projection of r onto the sun direction
    proj = np.einsum('ij,ij->i', r_2d, s_2d)         # (N,)

    # Perpendicular distance from the Earth–Sun line
    perp = r_2d - proj[:, np.newaxis] * s_2d          # (N, 3)
    perp_dist = np.linalg.norm(perp, axis=1)          # (N,)

    # In shadow when behind Earth (proj < 0) AND within the shadow cylinder
    lit = ~((proj < 0) & (perp_dist < body_radius))   # (N,)

    return lit[0] if scalar else lit
