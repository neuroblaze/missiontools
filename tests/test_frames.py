import numpy as np
import pytest

from missiontools.orbit.frames import gmst, eci_to_ecef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')
_ONE_DAY_US = np.timedelta64(86_400_000_000, 'us')
_SIDEREAL_DAY_US = np.timedelta64(int(86164.09053 * 1e6), 'us')

# GMST rate in degrees per Julian day (linear term of IAU 1982 formula)
GMST_RATE_DEG_PER_DAY = (876600 * 3600 + 8640184.812866) / 36525.0 / 240.0


# ---------------------------------------------------------------------------
# Reference value at J2000.0
# ---------------------------------------------------------------------------

def test_j2000_reference():
    """At J2000.0 GMST equals the IAU 1982 constant term:
    67310.54841 s = 280.46061837° (Astronomical Almanac reference value)."""
    expected = np.deg2rad(67310.54841 / 240.0) % (2 * np.pi)
    np.testing.assert_allclose(gmst(_J2000_US), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------

def test_output_in_0_to_2pi():
    """GMST must always lie in [0, 2π)."""
    times = _J2000_US + np.arange(500) * np.timedelta64(int(365.25 * 86400e6 / 499), 'us')
    result = gmst(times)
    assert np.all(result >= 0.0)
    assert np.all(result < 2 * np.pi)


# ---------------------------------------------------------------------------
# Sidereal periodicity
# ---------------------------------------------------------------------------

def test_sidereal_day_periodicity():
    """After exactly one sidereal day GMST must return to the same angle."""
    base_times = _J2000_US + np.array([0, 455, 1455], dtype='timedelta64[D]')
    np.testing.assert_allclose(
        gmst(base_times + _SIDEREAL_DAY_US) % (2 * np.pi),
        gmst(base_times) % (2 * np.pi),
        atol=1e-6,   # < 0.2 arcseconds
    )


# ---------------------------------------------------------------------------
# Drift rate
# ---------------------------------------------------------------------------

def test_gmst_rate_per_solar_day():
    """GMST should advance by ~360.985647° per Julian day (one solar day)."""
    delta_gmst = (gmst(_J2000_US + _ONE_DAY_US) - gmst(_J2000_US)) % (2 * np.pi)
    expected = np.deg2rad(GMST_RATE_DEG_PER_DAY % 360.0)
    np.testing.assert_allclose(delta_gmst, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Input flexibility
# ---------------------------------------------------------------------------

def test_scalar_datetime_input():
    """A scalar datetime64 should be accepted and return a scalar-like result."""
    result = gmst(_J2000_US)
    assert np.ndim(result) == 0 or isinstance(result, float)


def test_array_input_shape_preserved():
    """Output shape must match input shape."""
    times = _J2000_US + np.arange(37) * np.timedelta64(int(86400e6 / 36), 'us')
    assert gmst(times).shape == times.shape


# ===========================================================================
# eci_to_ecef
# ===========================================================================

# GMST Earth rotation rate in rad/s (used for Newton refinement)
_GMST_RATE_RAD_S = np.deg2rad(360.985647) / 86400.0


# ---------------------------------------------------------------------------
# Norm preservation
# ---------------------------------------------------------------------------

def test_eci_to_ecef_norm_preserved():
    """ECI→ECEF is a pure rotation; vector norms must be unchanged."""
    times = _J2000_US + np.arange(100) * np.timedelta64(600_000_000, 'us')
    r_eci = np.random.default_rng(0).standard_normal((100, 3)) * 7e6
    r_ecef = eci_to_ecef(r_eci, times)
    np.testing.assert_allclose(
        np.linalg.norm(r_ecef, axis=1),
        np.linalg.norm(r_eci, axis=1),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# z-component unchanged
# ---------------------------------------------------------------------------

def test_eci_to_ecef_z_unchanged():
    """Rotation about Z must not alter the z-component of any vector."""
    times = _J2000_US + np.arange(50) * np.timedelta64(3_600_000_000, 'us')
    r_eci = np.random.default_rng(1).standard_normal((50, 3)) * 7e6
    r_ecef = eci_to_ecef(r_eci, times)
    np.testing.assert_allclose(r_ecef[:, 2], r_eci[:, 2], rtol=1e-12)


# ---------------------------------------------------------------------------
# Identity at GMST = 0
# ---------------------------------------------------------------------------

def test_eci_to_ecef_identity_at_gmst_zero():
    """When GMST = 0 the ECI and ECEF frames are aligned: r_ECEF = r_ECI."""
    # GMST at J2000 ≈ 280.46°; need ~79.54° more → ~19 037 s to reach 360° ≡ 0°
    t0 = _J2000_US + np.timedelta64(19_037_000_000, 'us')
    # One Newton step to drive GMST to exactly 0
    theta0 = float(gmst(t0))
    dt_us = int(-theta0 / _GMST_RATE_RAD_S * 1e6)
    t_gmst0 = t0 + np.timedelta64(dt_us, 'us')

    r_eci = np.array([[7e6, 0.0, 0.0],
                      [0.0, 7e6, 0.0],
                      [0.0, 0.0, 7e6]])
    times = np.full(3, t_gmst0)
    r_ecef = eci_to_ecef(r_eci, times)
    np.testing.assert_allclose(r_ecef, r_eci, atol=1.0)  # < 1 m after Newton step


# ---------------------------------------------------------------------------
# Known rotation: GMST = π/2
# ---------------------------------------------------------------------------

def test_eci_to_ecef_known_rotation_quarter_turn():
    """At GMST = π/2 the ECI +x axis maps to ECEF (0, -1, 0)."""
    # Find a time when GMST ≈ π/2 using Newton refinement from J2000
    theta_j2000 = float(gmst(_J2000_US))           # ≈ 280.46° ≈ 4.894 rad
    target = np.pi / 2
    # Advance from J2000 until GMST would reach π/2 (unwrapped from 4.894 rad)
    # GMST must increase past 2π and then to π/2 (i.e. +target + (2π - theta_j2000))
    delta_s = (target + 2 * np.pi - theta_j2000) / _GMST_RATE_RAD_S
    t0 = _J2000_US + np.timedelta64(int(delta_s * 1e6), 'us')
    # Newton refinement
    theta0 = float(gmst(t0))
    dt_us = int(-(theta0 - target) / _GMST_RATE_RAD_S * 1e6)
    t_quarter = t0 + np.timedelta64(dt_us, 'us')

    r_eci = np.array([[1e7, 0.0, 0.0]])
    r_ecef = eci_to_ecef(r_eci, np.array([t_quarter]))
    np.testing.assert_allclose(r_ecef[0], [0.0, -1e7, 0.0], atol=1.0)


# ---------------------------------------------------------------------------
# Round-trip: ECI → ECEF → ECI
# ---------------------------------------------------------------------------

def test_eci_to_ecef_round_trip():
    """Applying the transpose (inverse rotation) must recover the original ECI vector."""
    times = _J2000_US + np.arange(20) * np.timedelta64(1_800_000_000, 'us')
    r_eci = np.random.default_rng(2).standard_normal((20, 3)) * 7e6
    r_ecef = eci_to_ecef(r_eci, times)

    # Inverse rotation Rz(+θ): negate sin terms relative to Rz(-θ)
    theta = np.atleast_1d(gmst(times))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(theta), np.ones_like(theta)
    Rz_inv = np.array([[cos_t, -sin_t, z],
                       [sin_t,  cos_t, z],
                       [    z,      z, o]]).transpose(2, 0, 1)
    r_eci_recovered = np.einsum('nij,nj->ni', Rz_inv, r_ecef)
    np.testing.assert_allclose(r_eci_recovered, r_eci, rtol=1e-12)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_eci_to_ecef_scalar_time_returns_1d():
    """Scalar time + (3,) vector should return shape (3,)."""
    result = eci_to_ecef(np.array([7e6, 0.0, 0.0]), _J2000_US)
    assert result.shape == (3,)


def test_eci_to_ecef_array_time_shape_preserved():
    """Output shape (N, 3) must match input for N-element time array."""
    n = 37
    times = _J2000_US + np.arange(n) * np.timedelta64(60_000_000, 'us')
    r_eci = np.random.default_rng(3).standard_normal((n, 3)) * 7e6
    assert eci_to_ecef(r_eci, times).shape == (n, 3)
