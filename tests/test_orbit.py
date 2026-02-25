import numpy as np
import pytest
from datetime import datetime, timedelta

from missiontools.orbit.propagation import propagate_analytical
from missiontools.orbit.constants import EARTH_MU

EPOCH = datetime(2025, 1, 1, 12, 0, 0)


def make_times(n, dt_seconds):
    return [EPOCH + timedelta(seconds=k * dt_seconds) for k in range(n)]


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A generic inclined elliptical orbit used by several tests
ELLIPTIC = dict(
    a=8_000_000.0,
    e=0.3,
    i=np.radians(45.0),
    arg_p=np.radians(90.0),
    raan=np.radians(60.0),
    ma=0.5,
)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    r, v = propagate_analytical(make_times(20, 60.0), EPOCH, **ELLIPTIC)
    assert r.shape == (20, 3)
    assert v.shape == (20, 3)


def test_single_timestep_shape():
    r, v = propagate_analytical([EPOCH], EPOCH, **ELLIPTIC)
    assert r.shape == (1, 3)
    assert v.shape == (1, 3)


# ---------------------------------------------------------------------------
# Circular orbit: radius is exactly a
# ---------------------------------------------------------------------------

def test_circular_orbit_constant_radius():
    """For e=0, |r| must equal a at every timestep (rotation preserves norms)."""
    a = 7_000_000.0
    r, _ = propagate_analytical(
        make_times(100, 60.0), EPOCH,
        a=a, e=0.0, i=np.radians(28.5),
        arg_p=0.0, raan=0.0, ma=0.0,
    )
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), a, rtol=1e-12)


# ---------------------------------------------------------------------------
# Vis-viva / specific orbital energy conservation
# ---------------------------------------------------------------------------

def test_energy_conservation():
    """ε = |v|²/2 − μ/|r| = −μ/(2a) must hold at every timestep."""
    a = ELLIPTIC["a"]
    r, v = propagate_analytical(make_times(200, 30.0), EPOCH, **ELLIPTIC)
    energy = 0.5 * np.sum(v**2, axis=1) - EARTH_MU / np.linalg.norm(r, axis=1)
    np.testing.assert_allclose(energy, -EARTH_MU / (2 * a), rtol=1e-10)


# ---------------------------------------------------------------------------
# Angular momentum conservation
# ---------------------------------------------------------------------------

def test_angular_momentum_conservation():
    """|r × v| = sqrt(μ · a · (1−e²)) must hold at every timestep."""
    a, e = ELLIPTIC["a"], ELLIPTIC["e"]
    r, v = propagate_analytical(make_times(200, 30.0), EPOCH, **ELLIPTIC)
    h_mag = np.linalg.norm(np.cross(r, v), axis=1)
    np.testing.assert_allclose(h_mag, np.sqrt(EARTH_MU * a * (1 - e**2)), rtol=1e-10)


# ---------------------------------------------------------------------------
# Orbit closes after one period
# ---------------------------------------------------------------------------

def test_orbit_closes_after_one_period():
    """After exactly one orbital period the state vector must repeat."""
    a, e = ELLIPTIC["a"], ELLIPTIC["e"]
    T = 2 * np.pi * np.sqrt(a**3 / EARTH_MU)
    times = [EPOCH, EPOCH + timedelta(seconds=T)]
    r, v = propagate_analytical(times, EPOCH, **ELLIPTIC)
    np.testing.assert_allclose(r[0], r[1], atol=1e-3)   # 1 mm
    np.testing.assert_allclose(v[0], v[1], atol=1e-6)   # 1 µm/s


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------

def test_perigee_position_along_eci_x():
    """At epoch with ma=0, i=0, raan=0, arg_p=0 the satellite is at
    perigee on the +x axis: r = (a(1−e), 0, 0)."""
    a, e = 8_000_000.0, 0.2
    r, _ = propagate_analytical(
        [EPOCH], EPOCH,
        a=a, e=e, i=0.0, arg_p=0.0, raan=0.0, ma=0.0,
    )
    np.testing.assert_allclose(r[0], [a * (1 - e), 0.0, 0.0], atol=1e-3)


def test_equatorial_orbit_zero_z():
    """For i=0 the z-component of every position vector must be zero."""
    r, _ = propagate_analytical(
        make_times(50, 120.0), EPOCH,
        a=7_000_000.0, e=0.05, i=0.0,
        arg_p=np.radians(45.0), raan=0.0, ma=0.5,
    )
    np.testing.assert_allclose(r[:, 2], 0.0, atol=1e-6)


def test_polar_orbit_crosses_z_axis():
    """For i=π/2 the angular momentum vector must lie in the xy-plane
    (h_z ≈ 0), i.e. the orbit passes through both poles."""
    r, v = propagate_analytical(
        make_times(100, 60.0), EPOCH,
        a=7_000_000.0, e=0.01, i=np.pi / 2,
        arg_p=0.0, raan=0.0, ma=0.0,
    )
    h = np.cross(r, v)
    np.testing.assert_allclose(h[:, 2], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_BASE = dict(
    epoch=EPOCH,
    a=7_000_000.0, e=0.01,
    i=np.radians(45.0),
    arg_p=0.5, raan=0.5, ma=0.5,
)


@pytest.mark.parametrize("override,match", [
    ({"a": 0.0},               "Semi-major axis"),
    ({"a": -1.0},              "Semi-major axis"),
    ({"e": -0.1},              "Eccentricity"),
    ({"e": 1.0},               "Eccentricity"),
    ({"i": -0.1},              "Inclination"),
    ({"i": np.pi + 0.1},       "Inclination"),
    ({"arg_p": -0.1},          "arg_p"),
    ({"arg_p": 2 * np.pi},     "arg_p"),
    ({"raan": -0.1},           "raan"),
    ({"raan": 2 * np.pi},      "raan"),
    ({"ma": -0.1},             "ma"),
    ({"ma": 2 * np.pi},        "ma"),
    ({"central_body_mu": 0.0}, "central_body_mu"),
    ({"central_body_mu":-1.0}, "central_body_mu"),
])
def test_input_validation_raises(override, match):
    kwargs = {**_BASE, **override}
    with pytest.raises(ValueError, match=match):
        propagate_analytical([EPOCH], **kwargs)
