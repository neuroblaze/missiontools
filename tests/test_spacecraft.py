import numpy as np
import pytest

from missiontools import Spacecraft
from missiontools.orbit import propagate_analytical, sun_synchronous_orbit
from missiontools.orbit.constants import EARTH_MU, EARTH_J2, EARTH_SEMI_MAJOR_AXIS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')

_KW = dict(
    a      = 6_771_000.0,
    e      = 0.0006,
    i      = np.radians(51.6),
    raan   = np.radians(120.0),
    arg_p  = np.radians(30.0),
    ma     = np.radians(0.0),
    epoch  = _EPOCH,
)

_EXPECTED_PARAMS_KEYS = {
    'epoch', 'a', 'e', 'i', 'raan', 'arg_p', 'ma',
    'central_body_mu', 'central_body_j2', 'central_body_radius',
}


# ===========================================================================
# Construction
# ===========================================================================

class TestSpacecraftConstruct:

    def test_construct_direct(self):
        sc = Spacecraft(**_KW)
        assert sc.a     == _KW['a']
        assert sc.e     == _KW['e']
        assert sc.i     == _KW['i']
        assert sc.raan  == _KW['raan']
        assert sc.arg_p == _KW['arg_p']
        assert sc.ma    == _KW['ma']

    def test_default_propagator_is_twobody(self):
        sc = Spacecraft(**_KW)
        assert sc.propagator_type == 'twobody'

    def test_propagator_j2(self):
        sc = Spacecraft(**_KW, propagator_type='j2')
        assert sc.propagator_type == 'j2'

    def test_invalid_propagator_raises(self):
        with pytest.raises(ValueError, match="propagator_type"):
            Spacecraft(**_KW, propagator_type='rk4')

    def test_default_central_body_earth(self):
        sc = Spacecraft(**_KW)
        assert sc.central_body_mu     == EARTH_MU
        assert sc.central_body_j2     == EARTH_J2
        assert sc.central_body_radius == EARTH_SEMI_MAJOR_AXIS

    def test_epoch_normalised_to_us_from_s(self):
        epoch_s = np.datetime64('2025-01-01T00:00:00', 's')
        sc = Spacecraft(**{**_KW, 'epoch': epoch_s})
        assert sc.epoch == _EPOCH

    def test_epoch_normalised_to_us_from_ms(self):
        epoch_ms = np.datetime64('2025-01-01T00:00:00.000', 'ms')
        sc = Spacecraft(**{**_KW, 'epoch': epoch_ms})
        assert sc.epoch == _EPOCH


# ===========================================================================
# keplerian_params property
# ===========================================================================

class TestKeplerianParams:

    def test_keys(self):
        sc = Spacecraft(**_KW)
        assert set(sc.keplerian_params.keys()) == _EXPECTED_PARAMS_KEYS

    def test_values_roundtrip(self):
        sc = Spacecraft(**_KW)
        p  = sc.keplerian_params
        assert p['a']     == sc.a
        assert p['e']     == sc.e
        assert p['i']     == sc.i
        assert p['raan']  == sc.raan
        assert p['arg_p'] == sc.arg_p
        assert p['ma']    == sc.ma
        assert p['epoch'] == sc.epoch
        assert p['central_body_mu']     == sc.central_body_mu
        assert p['central_body_j2']     == sc.central_body_j2
        assert p['central_body_radius'] == sc.central_body_radius


# ===========================================================================
# from_dict
# ===========================================================================

class TestFromDict:

    def test_from_dict_minimal(self):
        """7 required keys; central-body fields fall back to Earth defaults."""
        sc = Spacecraft.from_dict(_KW)
        assert sc.a    == _KW['a']
        assert sc.e    == _KW['e']
        assert sc.central_body_mu     == EARTH_MU
        assert sc.central_body_j2     == EARTH_J2
        assert sc.central_body_radius == EARTH_SEMI_MAJOR_AXIS

    def test_from_dict_full(self):
        """Full dict including optional central-body keys."""
        full = {**_KW,
                'central_body_mu':     1.23e14,
                'central_body_j2':     2.34e24,
                'central_body_radius': 6_400_000.0}
        sc = Spacecraft.from_dict(full)
        assert sc.central_body_mu     == 1.23e14
        assert sc.central_body_j2     == 2.34e24
        assert sc.central_body_radius == 6_400_000.0

    def test_from_dict_propagator_type(self):
        sc = Spacecraft.from_dict(_KW, propagator_type='j2')
        assert sc.propagator_type == 'j2'

    def test_from_dict_default_propagator(self):
        sc = Spacecraft.from_dict(_KW)
        assert sc.propagator_type == 'twobody'


# ===========================================================================
# Compatibility with functional API
# ===========================================================================

class TestFunctionalCompatibility:

    def test_compatible_with_propagate_analytical(self):
        sc = Spacecraft(**_KW)
        t  = np.array([_EPOCH, _EPOCH + np.timedelta64(90 * 60, 's')])
        r, v = propagate_analytical(t, **sc.keplerian_params)
        assert r.shape == (2, 3)
        assert v.shape == (2, 3)

    def test_compatible_with_propagate_analytical_j2(self):
        sc = Spacecraft(**_KW, propagator_type='j2')
        t  = np.array([_EPOCH, _EPOCH + np.timedelta64(90 * 60, 's')])
        r, v = propagate_analytical(t, **sc.keplerian_params, type=sc.propagator_type)
        assert r.shape == (2, 3)

    def test_from_dict_with_sun_synchronous_orbit(self):
        params = sun_synchronous_orbit(
            altitude           = 550_000.0,
            local_time_at_node = '10:30',
            node_type          = 'descending',
            epoch              = _EPOCH,
        )
        sc = Spacecraft.from_dict(params, propagator_type='j2')
        assert sc.e == pytest.approx(0.0, abs=1e-10)
        assert sc.propagator_type == 'j2'
        assert set(sc.keplerian_params.keys()) == _EXPECTED_PARAMS_KEYS


# ===========================================================================
# propagate
# ===========================================================================

_ONE_ORBIT = np.timedelta64(90 * 60, 's')   # ~1 ISS orbital period
_STEP_60S  = np.timedelta64(60, 's')


class TestPropagate:

    def test_output_keys(self):
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        assert set(result.keys()) == {'t', 'r', 'v'}

    def test_shapes(self):
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        N = len(result['t'])
        assert result['r'].shape == (N, 3)
        assert result['v'].shape == (N, 3)

    def test_t_start_and_end_included(self):
        """First sample must be t_start; last must be t_end."""
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        assert result['t'][0]  == np.asarray(_EPOCH,             dtype='datetime64[us]')
        assert result['t'][-1] == np.asarray(_EPOCH + _ONE_ORBIT, dtype='datetime64[us]')

    def test_step_spacing(self):
        """All consecutive samples must be separated by exactly one step."""
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        diffs  = np.diff(result['t'].astype('datetime64[us]').astype(np.int64))
        step_us = int(_STEP_60S / np.timedelta64(1, 'us'))
        # all gaps are one step except possibly the last (t_end clamp)
        assert (diffs[:-1] == step_us).all()

    def test_position_magnitude(self):
        """ECI position magnitude must be approximately a (semi-major axis)."""
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        r_mag  = np.linalg.norm(result['r'], axis=1)
        np.testing.assert_allclose(r_mag, sc.a, rtol=sc.e + 1e-4)

    def test_j2_propagator(self):
        """propagate() uses self.propagator_type without extra arguments."""
        sc     = Spacecraft(**_KW, propagator_type='j2')
        result = sc.propagate(_EPOCH, _EPOCH + _ONE_ORBIT, _STEP_60S)
        assert result['r'].shape[1] == 3

    def test_empty_window(self):
        """t_start == t_end returns empty arrays without raising."""
        sc     = Spacecraft(**_KW)
        result = sc.propagate(_EPOCH, _EPOCH, _STEP_60S)
        assert len(result['t']) == 0
        assert result['r'].shape == (0, 3)
        assert result['v'].shape == (0, 3)

    def test_non_us_epoch_inputs(self):
        """datetime64 inputs with non-µs resolution are handled correctly."""
        t_start = np.datetime64('2025-01-01T00:00:00', 's')
        t_end   = np.datetime64('2025-01-01T01:30:00', 's')
        sc      = Spacecraft(**_KW)
        result  = sc.propagate(t_start, t_end, _STEP_60S)
        assert result['t'].dtype == np.dtype('datetime64[us]')
        assert len(result['t']) > 0
