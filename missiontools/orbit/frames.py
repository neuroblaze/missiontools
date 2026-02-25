import numpy as np
import numpy.typing as npt

# J2000.0 epoch in UTC (≈ UT1 to < 1 s; TT leads UT1 by ~69 s and should NOT be used)
_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')
_SECONDS_PER_JULIAN_CENTURY = 36525.0 * 86400.0


def gmst(t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.floating]:
    """Greenwich Mean Sidereal Time (rad) from an array of UTC/UT1 datetimes.

    Uses the IAU 1982 polynomial. T is computed directly from the integer
    microsecond offset from J2000.0, avoiding the ~40 µs precision floor of
    a single-precision Julian Date float.

    Parameters
    ----------
    t : npt.NDArray[np.datetime64]
        Observation times as ``datetime64[us]``. Values are interpreted as
        **UT1** (passing UTC introduces < 0.004° error; passing TT introduces
        ~0.29° error and should be avoided).

    Returns
    -------
    npt.NDArray[np.floating]
        GMST in radians, wrapped to [0, 2π).
    """
    t_us = np.asarray(t, dtype='datetime64[us]')

    # Seconds from J2000.0 — int64 microsecond difference cast to float64
    # preserves ~0.1 µs precision (vs ~40 µs for a single JD float64)
    s = (t_us - _J2000_US).astype(np.float64) * 1e-6

    # Julian centuries from J2000.0
    T = s / _SECONDS_PER_JULIAN_CENTURY

    # IAU 1982 GMST polynomial — result in seconds of time
    theta = (67310.54841
             + (876600 * 3600 + 8640184.812866) * T
             + 0.093104 * T**2
             - 6.2e-6   * T**3)

    # Seconds of time → radians (1 s = 1/240 °), wrapped to [0, 2π)
    return (np.deg2rad(theta / 240.0)) % (2 * np.pi)

def eci_to_ecef(eci_vecs: npt.NDArray[np.floating],
                t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.floating]:
    """Convert ECI position/velocity vectors to ECEF via GMST rotation.

    Parameters
    ----------
    eci_vecs : npt.NDArray[np.floating]
        Vectors in the ECI frame, shape ``(N, 3)`` or ``(3,)`` for a single
        vector.
    t : npt.NDArray[np.datetime64]
        UTC/UT1 observation times as ``datetime64[us]``, shape ``(N,)`` or
        scalar. Must match the first dimension of ``eci_vecs``.

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in the ECEF frame, same shape as ``eci_vecs``.
    """
    theta = gmst(t)
    scalar = np.ndim(theta) == 0
    theta = np.atleast_1d(theta)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(theta), np.ones_like(theta)

    # Rz(-θ): ECI → ECEF rotates the frame eastward by the GMST angle
    Rz = np.array([[ cos_t,  sin_t, z],
                   [-sin_t,  cos_t, z],
                   [     z,      z, o]]).transpose(2, 0, 1)  # (N, 3, 3)

    result = np.einsum('nij,nj->ni', Rz, np.atleast_2d(eci_vecs))  # (N, 3)
    return result[0] if scalar else result