import numpy as np
import pytest

from missiontools.coverage import sample_aoi, sample_region, coverage_fraction, revisit_time

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Small rectangular polygon around the UK (lat/lon in radians)
# ~49°–59°N, ~8°W–2°E
_UK_POLY = np.radians([
    [49.0, -8.0],
    [59.0, -8.0],
    [59.0,  2.0],
    [49.0,  2.0],
])

# ISS-like orbit — epoch J2000
_J2000 = np.datetime64('2000-01-01T12:00:00', 'us')
_ISS = dict(
    epoch  = _J2000,
    a      = 6_771_000.0,
    e      = 0.0006,
    i      = np.radians(51.6),
    arg_p  = np.radians(30.0),
    raan   = np.radians(120.0),
    ma     = np.radians(0.0),
)

_T_START = _J2000
_T_END   = _J2000 + np.timedelta64(6 * 3600, 's')   # 6-hour window
_STEP    = np.timedelta64(60, 's')


# ===========================================================================
# sample_aoi
# ===========================================================================

class TestSampleAoi:

    def test_returns_two_arrays(self):
        lat, lon = sample_aoi(_UK_POLY, 50)
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert lat.shape == lon.shape

    def test_approximate_count(self):
        """Returned count should be within 20% of the requested n."""
        n = 200
        lat, lon = sample_aoi(_UK_POLY, n)
        assert 0.8 * n <= len(lat) <= 1.2 * n

    def test_all_points_inside_polygon(self):
        """Every returned point must lie inside the polygon."""
        from matplotlib.path import Path
        lat, lon = sample_aoi(_UK_POLY, 100)
        path = Path(_UK_POLY[:, ::-1])   # (lon, lat)
        inside = path.contains_points(np.column_stack([lon, lat]))
        assert inside.all(), f"{(~inside).sum()} point(s) outside polygon"

    def test_latitudes_in_range(self):
        lat, lon = sample_aoi(_UK_POLY, 50)
        lo, hi = _UK_POLY[:, 0].min(), _UK_POLY[:, 0].max()
        assert (lat >= lo).all() and (lat <= hi).all()

    def test_n_equals_1(self):
        lat, lon = sample_aoi(_UK_POLY, 1)
        assert len(lat) == 1
        assert len(lon) == 1

    def test_invalid_polygon_raises(self):
        with pytest.raises(ValueError, match="polygon"):
            sample_aoi(np.zeros((4, 3)), 10)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            sample_aoi(_UK_POLY, 0)


# ===========================================================================
# coverage_fraction
# ===========================================================================

def _sample_uk(n=30):
    return sample_aoi(_UK_POLY, n)


class TestCoverageFraction:

    def test_output_keys(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        for key in ('t', 'fraction', 'cumulative', 'mean_fraction',
                    'final_cumulative'):
            assert key in result

    def test_array_shapes(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        N = len(result['t'])
        assert N > 0
        assert result['fraction'].shape  == (N,)
        assert result['cumulative'].shape == (N,)

    def test_fraction_in_unit_interval(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert (result['fraction']  >= 0.0).all()
        assert (result['fraction']  <= 1.0).all()
        assert (result['cumulative'] >= 0.0).all()
        assert (result['cumulative'] <= 1.0).all()

    def test_cumulative_non_decreasing(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        diffs = np.diff(result['cumulative'])
        assert (diffs >= -1e-6).all(), "cumulative coverage must not decrease"

    def test_mean_fraction_consistent(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert abs(result['mean_fraction'] - result['fraction'].mean()) < 1e-4

    def test_final_cumulative_equals_last_element(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert abs(result['final_cumulative'] - result['cumulative'][-1]) < 1e-6

    def test_timestamps_are_datetime64(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert result['t'].dtype == np.dtype('datetime64[us]')
        assert result['t'][0]  == _T_START
        assert result['t'][-1] == _T_END

    def test_empty_window_returns_empty(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_START,
                                   max_step=_STEP)
        assert len(result['t']) == 0

    def test_batching_consistent(self):
        """Result must be identical regardless of batch_size."""
        lat, lon = _sample_uk(20)
        kwargs   = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                        max_step=_STEP)
        r_small = coverage_fraction(lat, lon, batch_size=5,    **kwargs)
        r_large = coverage_fraction(lat, lon, batch_size=5000, **kwargs)
        np.testing.assert_array_equal(r_small['fraction'],  r_large['fraction'])
        np.testing.assert_array_equal(r_small['cumulative'], r_large['cumulative'])

    def test_j2_propagator_runs(self):
        lat, lon = _sample_uk(10)
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   propagator_type='j2', max_step=_STEP)
        assert isinstance(result['mean_fraction'], float)


# ===========================================================================
# revisit_time
# ===========================================================================

class TestRevisitTime:

    def test_output_keys(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        for key in ('max_revisit', 'mean_revisit', 'global_max', 'global_mean'):
            assert key in result

    def test_shapes(self):
        lat, lon = _sample_uk(20)
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        assert result['max_revisit'].shape  == (20,)
        assert result['mean_revisit'].shape == (20,)

    def test_max_ge_mean(self):
        """Per-point max revisit must be ≥ mean revisit (ignoring NaN)."""
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        mx = result['max_revisit']
        mn = result['mean_revisit']
        valid = ~(np.isnan(mx) | np.isnan(mn))
        assert (mx[valid] >= mn[valid] - 1e-6).all()

    def test_positive_revisit_times(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        for arr in (result['max_revisit'], result['mean_revisit']):
            valid = arr[~np.isnan(arr)]
            assert (valid > 0).all()

    def test_global_max_ge_all_per_point(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        if not np.isnan(result['global_max']):
            assert result['global_max'] >= np.nanmax(result['max_revisit']) - 1e-6

    def test_empty_window_returns_nan(self):
        lat, lon = _sample_uk(5)
        result = revisit_time(lat, lon, _ISS, _T_START, _T_START,
                              max_step=_STEP)
        assert np.isnan(result['global_max'])
        assert np.isnan(result['global_mean'])

    def test_batching_consistent(self):
        """Revisit statistics must be identical regardless of batch_size."""
        lat, lon = _sample_uk(15)
        kwargs   = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                        max_step=_STEP)
        r_small = revisit_time(lat, lon, batch_size=5,    **kwargs)
        r_large = revisit_time(lat, lon, batch_size=5000, **kwargs)
        np.testing.assert_array_equal(r_small['max_revisit'],  r_large['max_revisit'])
        np.testing.assert_array_equal(r_small['mean_revisit'], r_large['mean_revisit'])


# ===========================================================================
# sample_region
# ===========================================================================

class TestSampleRegion:

    # --- point count scales with area ---

    def test_global_returns_points(self):
        """All-None call must return a non-empty global sample."""
        lat, lon = sample_region()
        assert len(lat) > 0
        assert len(lat) == len(lon)

    def test_denser_density_gives_more_points(self):
        """Halving point_density should roughly double the point count."""
        lat_a, _ = sample_region(point_density=2e11)
        lat_b, _ = sample_region(point_density=1e11)
        assert len(lat_b) > len(lat_a)

    # --- latitude bounds ---

    def test_lat_band_respects_bounds(self):
        """All returned latitudes must lie within [lat_min, lat_max]."""
        lo, hi = np.radians(30.0), np.radians(60.0)
        lat, _ = sample_region(lat_min=lo, lat_max=hi, point_density=1e11)
        assert (lat >= lo).all() and (lat <= hi).all()

    def test_lat_min_none_reaches_south_pole(self):
        """lat_min=None must allow points below −60°."""
        lat, _ = sample_region(lat_max=np.radians(0.0), point_density=5e12)
        assert lat.min() < np.radians(-60.0)

    def test_lat_max_none_reaches_north_pole(self):
        """lat_max=None must allow points above +60°."""
        lat, _ = sample_region(lat_min=np.radians(0.0), point_density=5e12)
        assert lat.max() > np.radians(60.0)

    # --- longitude bounds ---

    def test_lon_band_respects_bounds(self):
        """All returned longitudes must lie within [lon_min, lon_max]."""
        lo, hi = np.radians(-10.0), np.radians(40.0)
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=1e11)
        assert (lon >= lo).all() and (lon <= hi).all()

    def test_antimeridian_no_points_in_gap(self):
        """For a crossing region (lon_min > lon_max) no point should fall
        in the excluded middle band."""
        lo, hi = np.radians(160.0), np.radians(-160.0)   # 320° span
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=2e11)
        # Gap is (hi, lo) = (-160°, 160°); no point should be in there
        in_gap = (lon > hi) & (lon < lo)
        assert not in_gap.any()

    def test_antimeridian_has_points_on_both_sides(self):
        """A crossing region should contain points east and west of ±180°."""
        lo, hi = np.radians(150.0), np.radians(-150.0)
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=5e11)
        assert (lon >= lo).any(), "no points east of antimeridian"
        assert (lon <= hi).any(), "no points west of antimeridian"

    # --- all-None shorthand ---

    def test_all_none_is_global(self):
        """sample_region() with no arguments should cover all latitudes."""
        lat, lon = sample_region(point_density=5e12)
        assert lat.min() < np.radians(-60.0)
        assert lat.max() > np.radians(60.0)
        # Longitudes should span close to full circle
        assert lon.max() - lon.min() > np.radians(300.0)

    # --- validation ---

    def test_mismatched_lon_raises(self):
        with pytest.raises(ValueError, match="lon_min and lon_max"):
            sample_region(lon_min=np.radians(0.0))

    def test_mismatched_lon_raises_other_direction(self):
        with pytest.raises(ValueError, match="lon_min and lon_max"):
            sample_region(lon_max=np.radians(0.0))

    def test_inverted_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            sample_region(lat_min=np.radians(60.0), lat_max=np.radians(30.0))

    def test_equal_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            sample_region(lat_min=np.radians(45.0), lat_max=np.radians(45.0))

    def test_nonpositive_density_raises(self):
        with pytest.raises(ValueError, match="point_density"):
            sample_region(point_density=0.0)

    def test_nonpositive_density_negative_raises(self):
        with pytest.raises(ValueError, match="point_density"):
            sample_region(point_density=-1e10)


# ===========================================================================
# FOV cone constraint
# ===========================================================================

# Nadir pointing in LVLH = -R̂ direction (negative radial)
_NADIR_LVLH = np.array([-1.0, 0.0, 0.0])


class TestCoverageFractionFov:

    def test_wide_nadir_fov_matches_no_fov(self):
        """90° nadir half-angle covers the whole forward hemisphere — should
        equal or nearly equal the horizon-only result."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_fov  = coverage_fraction(lat, lon, **kw,
                                   fov_pointing_lvlh=_NADIR_LVLH,
                                   fov_half_angle=np.radians(90.0))
        np.testing.assert_array_equal(r_base['fraction'], r_fov['fraction'])

    def test_narrow_fov_reduces_coverage(self):
        """A 20° nadir FOV must produce ≤ coverage than no FOV constraint."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_fov  = coverage_fraction(lat, lon, **kw,
                                   fov_pointing_lvlh=_NADIR_LVLH,
                                   fov_half_angle=np.radians(20.0))
        assert r_fov['mean_fraction'] <= r_base['mean_fraction'] + 1e-6
        assert r_fov['final_cumulative'] <= r_base['final_cumulative'] + 1e-6

    def test_pointing_unnormalised_matches_normalised(self):
        """Passing a non-unit pointing vector must give the same result as a
        pre-normalised one."""
        lat, lon = _sample_uk(15)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP, fov_half_angle=np.radians(30.0))
        r_norm   = coverage_fraction(lat, lon, **kw,
                                     fov_pointing_lvlh=_NADIR_LVLH)
        r_scaled = coverage_fraction(lat, lon, **kw,
                                     fov_pointing_lvlh=_NADIR_LVLH * 5.0)
        np.testing.assert_array_equal(r_norm['fraction'], r_scaled['fraction'])

    def test_fov_missing_angle_raises(self):
        """Providing only fov_pointing_lvlh must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP,
                              fov_pointing_lvlh=_NADIR_LVLH)

    def test_fov_missing_pointing_raises(self):
        """Providing only fov_half_angle must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP,
                              fov_half_angle=np.radians(30.0))


class TestRevisitTimeFov:

    def test_fov_reduces_or_equals_revisit(self):
        """With a narrow FOV each point is visited less often, so global_mean
        revisit time must be ≥ the unconstrained case (or NaN if never seen)."""
        lat, lon = _sample_uk(15)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = revisit_time(lat, lon, **kw)
        r_fov  = revisit_time(lat, lon, **kw,
                              fov_pointing_lvlh=_NADIR_LVLH,
                              fov_half_angle=np.radians(20.0))
        base_mean = r_base['global_mean']
        fov_mean  = r_fov['global_mean']
        if not np.isnan(fov_mean) and not np.isnan(base_mean):
            assert fov_mean >= base_mean - 1e-6

    def test_revisit_fov_missing_angle_raises(self):
        """Providing only fov_pointing_lvlh must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            revisit_time(lat, lon, _ISS, _T_START, _T_END,
                         max_step=_STEP,
                         fov_pointing_lvlh=_NADIR_LVLH)
