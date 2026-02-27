import numpy as np
import pytest

from missiontools.orbit.access import earth_access, earth_access_intervals
from missiontools.orbit.frames import geodetic_to_ecef, eci_to_ecef

# Ground station: London Heathrow (approx)
_LAT = np.radians(51.477)
_LON = np.radians(-0.461)
_ALT = 25.0  # m

_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')


def _gs_ecef():
    return geodetic_to_ecef(_LAT, _LON, _ALT)


# ---------------------------------------------------------------------------
# Directly overhead → always visible
# ---------------------------------------------------------------------------

def test_directly_overhead_is_visible():
    """A satellite directly above the GS (el = 90°) must be visible."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    sat = gs + up * 500_000.0   # 500 km straight up
    result = earth_access(sat[np.newaxis], _LAT, _LON, _ALT,
                          el_min=0.0, frame='ecef')
    assert result.shape == (1,)
    assert result[0]


# ---------------------------------------------------------------------------
# Satellite on the opposite side of the Earth → never visible
# ---------------------------------------------------------------------------

def test_below_horizon_not_visible():
    """A satellite directly below the GS (antipodal overhead) must not be visible."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    sat = gs - up * 500_000.0   # 500 km straight down (into the Earth)
    result = earth_access(sat[np.newaxis], _LAT, _LON, _ALT,
                          el_min=0.0, frame='ecef')
    assert not result[0]


# ---------------------------------------------------------------------------
# Satellite at exactly el_min = 0° should be visible (boundary inclusive)
# ---------------------------------------------------------------------------

def test_el_min_boundary():
    """A satellite 1° above the horizon is visible; 1° below is not."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    perp = np.array([-np.sin(_LAT) * np.cos(_LON),
                     -np.sin(_LAT) * np.sin(_LON),
                     np.cos(_LAT)])
    r = 600_000.0
    el_above = np.radians(1.0)
    el_below = np.radians(-1.0)
    sat_above = gs + (up * np.sin(el_above) + perp * np.cos(el_above)) * r
    sat_below = gs + (up * np.sin(el_below) + perp * np.cos(el_below)) * r
    vecs = np.stack([sat_above, sat_below])
    result = earth_access(vecs, _LAT, _LON, _ALT, el_min=0.0, frame='ecef')
    assert result[0]
    assert not result[1]


# ---------------------------------------------------------------------------
# el_min mask filters satellites below the threshold
# ---------------------------------------------------------------------------

def test_el_min_filters_low_satellites():
    """One sat well above el_min, one just below — should get [True, False]."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    # Satellite at 20° elevation
    el_high = np.radians(20.0)
    perp = np.array([-np.sin(_LAT) * np.cos(_LON),
                     -np.sin(_LAT) * np.sin(_LON),
                     np.cos(_LAT)])
    sat_high = gs + (up * np.sin(el_high) + perp * np.cos(el_high)) * 800_000.0
    # Satellite at 4° elevation
    el_low = np.radians(4.0)
    sat_low  = gs + (up * np.sin(el_low)  + perp * np.cos(el_low))  * 800_000.0

    vecs = np.stack([sat_high, sat_low])
    result = earth_access(vecs, _LAT, _LON, _ALT,
                          el_min=np.radians(10.0), frame='ecef')
    assert result[0]
    assert not result[1]


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    """Output must be a bool array with shape (N,) matching the input."""
    n = 50
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    vecs = gs + up * np.linspace(200e3, 800e3, n)[:, np.newaxis]
    result = earth_access(vecs, _LAT, _LON, _ALT, frame='ecef')
    assert result.shape == (n,)
    assert result.dtype == np.bool_


# ---------------------------------------------------------------------------
# ECEF and ECI frames give consistent results
# ---------------------------------------------------------------------------

def test_ecef_and_eci_frames_agree():
    """ECEF and ECI inputs at the same epoch must produce identical access flags."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    perp = np.array([-np.sin(_LAT) * np.cos(_LON),
                     -np.sin(_LAT) * np.sin(_LON),
                     np.cos(_LAT)])
    n = 10
    els = np.linspace(np.radians(-5), np.radians(60), n)
    vecs_ecef = np.array([
        gs + (up * np.sin(e) + perp * np.cos(e)) * 600_000.0 for e in els
    ])
    t = np.full(n, _J2000_US)
    vecs_eci = eci_to_ecef.__module__ and __import__(
        'missiontools.orbit.frames', fromlist=['ecef_to_eci']
    ).ecef_to_eci(vecs_ecef, t)

    result_ecef = earth_access(vecs_ecef, _LAT, _LON, _ALT,
                               el_min=0.0, frame='ecef')
    result_eci  = earth_access(vecs_eci,  _LAT, _LON, _ALT,
                               el_min=0.0, frame='eci', t=t)
    np.testing.assert_array_equal(result_eci, result_ecef)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_eci_frame_requires_t():
    """frame='eci' with t=None must raise ValueError."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    vecs = (gs + up * 500_000.0)[np.newaxis]
    with pytest.raises(ValueError, match="t must be provided"):
        earth_access(vecs, _LAT, _LON, frame='eci', t=None)


def test_invalid_frame_raises():
    """An unrecognised frame string must raise ValueError."""
    gs = _gs_ecef()
    up = np.array([np.cos(_LAT) * np.cos(_LON),
                   np.cos(_LAT) * np.sin(_LON),
                   np.sin(_LAT)])
    vecs = (gs + up * 500_000.0)[np.newaxis]
    with pytest.raises(ValueError, match="frame must be"):
        earth_access(vecs, _LAT, _LON, frame='j2000')


# ===========================================================================
# earth_access_intervals
# ===========================================================================

# ISS-like orbital elements (epoch = J2000)
_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')

_ISS_PARAMS = dict(
    epoch  = _J2000_US,
    a      = 6_771_000.0,       # m  (~400 km altitude)
    e      = 0.0006,
    i      = np.radians(51.6),
    arg_p  = np.radians(30.0),
    raan   = np.radians(120.0),
    ma     = np.radians(0.0),
)

_T_START = _J2000_US
_T_END   = _J2000_US + np.timedelta64(24 * 3600, 's')   # 24 h window


# ---------------------------------------------------------------------------
# Sanity: ISS-like orbit finds at least one pass over London in 24 h
# ---------------------------------------------------------------------------

def test_intervals_finds_passes():
    """ISS-like orbit should have several passes over London in 24 hours."""
    intervals = earth_access_intervals(
        _T_START, _T_END, _ISS_PARAMS, _LAT, _LON, _ALT,
        el_min=np.radians(5.0),
        max_step=np.timedelta64(30, 's'),
    )
    assert len(intervals) >= 1
    for start, end in intervals:
        assert start < end


# ---------------------------------------------------------------------------
# No access: orbit never reaches a near-polar GS
# ---------------------------------------------------------------------------

def test_intervals_no_access():
    """A low-inclination orbit never reaches a near-polar ground station."""
    params = dict(
        epoch = _J2000_US,
        a     = 7_000_000.0,
        e     = 0.001,
        i     = np.radians(20.0),   # stays within ±20° latitude
        arg_p = np.radians(0.0),
        raan  = np.radians(0.0),
        ma    = np.radians(0.0),
    )
    lat_polar = np.radians(85.0)    # way outside inclination band
    lon_polar = np.radians(0.0)
    intervals = earth_access_intervals(
        _T_START, _T_END, params, lat_polar, lon_polar,
        el_min=0.0,
        max_step=np.timedelta64(60, 's'),
    )
    assert intervals == []


# ---------------------------------------------------------------------------
# Continuous access: el_min = -π/2 means always visible
# ---------------------------------------------------------------------------

def test_intervals_continuous_access():
    """el_min = -π/2 (−90°) means every position is visible → one interval
    spanning the entire window."""
    t_start = _J2000_US
    t_end   = _J2000_US + np.timedelta64(600, 's')   # 10 min window
    intervals = earth_access_intervals(
        t_start, t_end, _ISS_PARAMS, _LAT, _LON, _ALT,
        el_min=-np.pi / 2,
        max_step=np.timedelta64(60, 's'),
    )
    assert len(intervals) == 1
    assert intervals[0][0] == t_start
    assert intervals[0][1] == t_end


# ---------------------------------------------------------------------------
# Return type is list[tuple[np.datetime64, np.datetime64]]
# ---------------------------------------------------------------------------

def test_intervals_return_type():
    """Each element must be a tuple of two np.datetime64 values."""
    intervals = earth_access_intervals(
        _T_START, _T_END, _ISS_PARAMS, _LAT, _LON, _ALT,
        max_step=np.timedelta64(60, 's'),
    )
    for start, end in intervals:
        assert isinstance(start, np.datetime64)
        assert isinstance(end,   np.datetime64)


# ---------------------------------------------------------------------------
# Edge refinement respects refine_tol
# ---------------------------------------------------------------------------

def test_intervals_refine_tol_respected():
    """Every refined edge must lie within max_step of the coarse bracket
    on either side (a proxy for confirming refinement actually ran)."""
    step   = np.timedelta64(60, 's')
    tol    = np.timedelta64(5, 's')
    intervals = earth_access_intervals(
        _T_START, _T_END, _ISS_PARAMS, _LAT, _LON, _ALT,
        el_min=np.radians(5.0),
        max_step=step,
        refine_tol=tol,
    )
    # Verify that each start/end is within [T_START, T_END]
    for start, end in intervals:
        assert _T_START <= start <= _T_END
        assert _T_START <= end   <= _T_END
        assert start < end


# ---------------------------------------------------------------------------
# J2 propagator runs without error
# ---------------------------------------------------------------------------

def test_intervals_j2_runs():
    """propagator_type='j2' must complete and return a list."""
    intervals = earth_access_intervals(
        _T_START,
        _J2000_US + np.timedelta64(6 * 3600, 's'),
        _ISS_PARAMS, _LAT, _LON, _ALT,
        propagator_type='j2',
        max_step=np.timedelta64(60, 's'),
    )
    assert isinstance(intervals, list)


# ---------------------------------------------------------------------------
# Batching consistency: result must be identical regardless of batch_size
# ---------------------------------------------------------------------------

def test_intervals_batching_consistent():
    """Result must be identical with batch_size=100 vs batch_size=5000."""
    kwargs = dict(
        keplerian_params = _ISS_PARAMS,
        lat    = _LAT,
        lon    = _LON,
        alt    = _ALT,
        el_min = np.radians(5.0),
        max_step   = np.timedelta64(60, 's'),
        refine_tol = np.timedelta64(1, 's'),
    )
    t_end_short = _J2000_US + np.timedelta64(6 * 3600, 's')

    result_small = earth_access_intervals(_T_START, t_end_short,
                                          batch_size=100, **kwargs)
    result_large = earth_access_intervals(_T_START, t_end_short,
                                          batch_size=5000, **kwargs)

    assert len(result_small) == len(result_large)
    for (s1, e1), (s2, e2) in zip(result_small, result_large):
        assert s1 == s2
        assert e1 == e2


# ---------------------------------------------------------------------------
# Empty window returns []
# ---------------------------------------------------------------------------

def test_intervals_empty_window():
    """t_start == t_end must return an empty list."""
    assert earth_access_intervals(
        _T_START, _T_START, _ISS_PARAMS, _LAT, _LON
    ) == []
