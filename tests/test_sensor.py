import numpy as np
import pytest

from missiontools import (
    Spacecraft,
    FixedAttitudeLaw,
    TrackAttitudeLaw,
    AbstractAttitudeLaw,
    AbstractSensor,
    ConicSensor,
    RectangularSensor,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64("2025-01-01T00:00:00", "us")

_KW = dict(
    a=6_771_000.0,
    e=0.0,
    i=np.radians(51.6),
    raan=0.0,
    arg_p=0.0,
    ma=0.0,
    epoch=_EPOCH,
)


def _sc():
    return Spacecraft(**_KW)


def _orbit_state(sc):
    """Return a single ECI state (r, v, t) for testing pointing methods."""
    state = sc.propagate(
        _EPOCH, _EPOCH + np.timedelta64(60, "s"), np.timedelta64(60, "s")
    )
    return state["r"][0], state["v"][0], state["t"][0]


# ===========================================================================
# Construction and validation
# ===========================================================================


class TestSensorConstruct:
    def test_no_mode_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            ConicSensor(10.0)

    def test_two_modes_raises(self):
        with pytest.raises(ValueError, match="Only one"):
            ConicSensor(
                10.0, attitude_law=FixedAttitudeLaw.nadir(), body_vector=[0, 0, 1]
            )

    def test_independent_mode_stored(self):
        law = FixedAttitudeLaw.nadir()
        s = ConicSensor(10.0, attitude_law=law)
        assert s._mode == "independent"
        assert s._attitude_law is law

    def test_body_vector_mode_stored(self):
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        assert s._mode == "body"

    def test_body_euler_mode_stored(self):
        s = ConicSensor(10.0, body_euler_deg=(0, 0, 0))
        assert s._mode == "body"

    def test_invalid_attitude_law_type_raises(self):
        with pytest.raises(TypeError, match="AbstractAttitudeLaw"):
            ConicSensor(10.0, attitude_law="nadir")


class TestSensorHalfAngle:
    def test_stored_as_radians(self):
        s = ConicSensor(30.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.half_angle_rad, np.radians(30.0))

    def test_90_deg_accepted(self):
        s = ConicSensor(90.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.half_angle_rad, np.pi / 2)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="half_angle_deg"):
            ConicSensor(0.0, body_vector=[0, 0, 1])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="half_angle_deg"):
            ConicSensor(-5.0, body_vector=[0, 0, 1])

    def test_above_90_raises(self):
        with pytest.raises(ValueError, match="half_angle_deg"):
            ConicSensor(91.0, body_vector=[0, 0, 1])


# ===========================================================================
# Body-vector mode
# ===========================================================================


class TestSensorBodyVector:
    def test_body_vector_is_unit(self):
        s = ConicSensor(10.0, body_vector=[3, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(s._body_vector), 1.0)

    def test_body_vector_non_unit_normalised(self):
        s = ConicSensor(10.0, body_vector=[0, 0, 5])
        np.testing.assert_allclose(s._body_vector, [0, 0, 1])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            ConicSensor(10.0, body_vector=[0, 0, 0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ConicSensor(10.0, body_vector=[1, 0])

    def test_pointing_raises_before_attach(self):
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        sc = _sc()
        r, v, t = _orbit_state(sc)
        with pytest.raises(RuntimeError, match="add_sensor"):
            s.pointing_eci(r, v, t)

    def test_spacecraft_none_before_attach(self):
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        assert s.spacecraft is None


# ===========================================================================
# Body-euler mode
# ===========================================================================


class TestSensorBodyEuler:
    def test_identity_euler_gives_body_z(self):
        s = ConicSensor(10.0, body_euler_deg=(0, 0, 0))
        np.testing.assert_allclose(s._body_vector, [0, 0, 1], atol=1e-12)

    def test_pitch_90_gives_negative_body_x(self):
        # 90° pitch (yaw=0, pitch=90, roll=0): sensor-z → body-x neg direction
        s = ConicSensor(10.0, body_euler_deg=(0, 90, 0))
        np.testing.assert_allclose(s._body_vector, [-1, 0, 0], atol=1e-12)

    def test_equivalent_to_body_vector(self):
        # Identity euler → same as body_vector=[0,0,1]
        s_euler = ConicSensor(10.0, body_euler_deg=(0, 0, 0))
        s_vector = ConicSensor(10.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(
            s_euler._body_vector, s_vector._body_vector, atol=1e-12
        )

    def test_yaw_only_does_not_change_boresight(self):
        # Yaw rotates about body-z — does not move the z-axis
        s0 = ConicSensor(10.0, body_euler_deg=(0, 0, 0))
        s1 = ConicSensor(10.0, body_euler_deg=(45, 0, 0))
        s2 = ConicSensor(10.0, body_euler_deg=(90, 0, 0))
        np.testing.assert_allclose(s0._body_vector, s1._body_vector, atol=1e-12)
        np.testing.assert_allclose(s0._body_vector, s2._body_vector, atol=1e-12)


# ===========================================================================
# Independent mode
# ===========================================================================


class TestSensorIndependent:
    def test_stores_attitude_law(self):
        law = FixedAttitudeLaw.nadir()
        s = ConicSensor(10.0, attitude_law=law)
        assert s._attitude_law is law

    def test_pointing_eci_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s = ConicSensor(10.0, attitude_law=law)
        sc = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_eci(r, v, t),
            law.pointing_eci(r, v, t),
            atol=1e-12,
        )

    def test_pointing_lvlh_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s = ConicSensor(10.0, attitude_law=law)
        sc = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_lvlh(r, v, t),
            law.pointing_lvlh(r, v, t),
            atol=1e-12,
        )

    def test_pointing_ecef_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s = ConicSensor(10.0, attitude_law=law)
        sc = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_ecef(r, v, t),
            law.pointing_ecef(r, v, t),
            atol=1e-12,
        )

    def test_no_spacecraft_needed(self):
        # Independent sensors work without an attached spacecraft
        s = ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir())
        assert s.spacecraft is None
        sc = _sc()
        r, v, t = _orbit_state(sc)
        result = s.pointing_eci(r, v, t)
        assert result.shape == (3,)


# ===========================================================================
# Spacecraft–sensor relationship
# ===========================================================================


class TestSpacecraftSensorRelationship:
    def test_sensors_empty_by_default(self):
        assert _sc().sensors == []

    def test_add_sensor_grows_list(self):
        sc = _sc()
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)
        assert len(sc.sensors) == 1

    def test_back_reference_set(self):
        sc = _sc()
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)
        assert s.spacecraft is sc

    def test_sensors_returns_copy(self):
        sc = _sc()
        sc.add_sensor(ConicSensor(10.0, body_vector=[0, 0, 1]))
        lst = sc.sensors
        lst.clear()
        assert len(sc.sensors) == 1  # original unaffected

    def test_multiple_sensors_stored(self):
        sc = _sc()
        s1 = ConicSensor(10.0, body_vector=[0, 0, 1])
        s2 = ConicSensor(20.0, body_vector=[0, 1, 0])
        sc.add_sensor(s1)
        sc.add_sensor(s2)
        assert len(sc.sensors) == 2
        assert sc.sensors[0] is s1
        assert sc.sensors[1] is s2

    def test_wrong_type_raises(self):
        sc = _sc()
        with pytest.raises(TypeError, match="AbstractSensor"):
            sc.add_sensor("not_a_sensor")


# ===========================================================================
# Pointing correctness
# ===========================================================================


class TestSensorPointing:
    """Verify body-mounted sensor pointing directions on a nadir spacecraft.

    Nadir spacecraft (FixedAttitudeLaw.nadir()) quaternion maps body frame → LVLH:
      body-z [0,0,1] → LVLH [-1, 0, 0]  (nadir = −R̂)
      body-x [1,0,0] → LVLH [ 0, 1, 0]  (along-track = Ŝ)
      body-y [0,1,0] → LVLH [ 0, 0,-1]  (−orbit-normal = −Ŵ)
    """

    def _setup(self, body_vec):
        sc = _sc()  # nadir spacecraft by default
        s = ConicSensor(10.0, body_vector=body_vec)
        sc.add_sensor(s)
        r, v, t = _orbit_state(sc)
        return s, r, v, t

    def test_body_z_sensor_on_nadir_sc_points_nadir(self):
        s, r, v, t = self._setup([0, 0, 1])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_body_x_sensor_on_nadir_sc_points_along_track(self):
        s, r, v, t = self._setup([1, 0, 0])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [0.0, 1.0, 0.0], atol=1e-10)

    def test_body_y_sensor_on_nadir_sc_points_minus_orbit_normal(self):
        s, r, v, t = self._setup([0, 1, 0])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [0.0, 0.0, -1.0], atol=1e-10)

    def test_pointing_eci_is_unit(self):
        s, r, v, t = self._setup([0, 0, 1])
        np.testing.assert_allclose(
            np.linalg.norm(s.pointing_eci(r, v, t)), 1.0, atol=1e-12
        )

    def test_pointing_lvlh_is_unit(self):
        s, r, v, t = self._setup([0, 1, 0])
        np.testing.assert_allclose(
            np.linalg.norm(s.pointing_lvlh(r, v, t)), 1.0, atol=1e-12
        )

    def test_pointing_ecef_is_unit(self):
        s, r, v, t = self._setup([1, 0, 0])
        np.testing.assert_allclose(
            np.linalg.norm(s.pointing_ecef(r, v, t)), 1.0, atol=1e-12
        )

    def test_body_mounted_sensor_on_tracking_sc_boresight_matches(self):
        """Body-z sensor on a tracking spacecraft must point at the target."""
        sc_host = _sc()
        sc_target = Spacecraft(
            a=7_000_000.0,
            e=0.0,
            i=np.radians(60.0),
            raan=0.5,
            arg_p=0.0,
            ma=0.0,
            epoch=_EPOCH,
        )
        sc_host.attitude_law = TrackAttitudeLaw(sc_target)
        s = ConicSensor(10.0, body_vector=[0, 0, 1])  # sensor boresight = body-z
        sc_host.add_sensor(s)

        r, v, t = _orbit_state(sc_host)
        sensor_pointing = s.pointing_eci(r, v, t)
        law_pointing = sc_host.attitude_law.pointing_eci(r, v, t)
        np.testing.assert_allclose(sensor_pointing, law_pointing, atol=1e-10)


# ===========================================================================
# AbstractSensor hierarchy
# ===========================================================================


class TestAbstractSensor:
    def test_conic_is_subclass(self):
        assert issubclass(ConicSensor, AbstractSensor)

    def test_abstract_sensor_not_instantiable(self):
        with pytest.raises(TypeError):
            AbstractSensor()

    def test_conic_instance_passes_isinstance(self):
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        assert isinstance(s, AbstractSensor)

    def test_spacecraft_accepts_conic_via_abstract_guard(self):
        sc = _sc()
        s = ConicSensor(10.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)  # must not raise
        assert s.spacecraft is sc


# ===========================================================================
# RectangularSensor — construction and validation
# ===========================================================================


class TestRectangularSensorConstruct:
    def test_no_mode_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            RectangularSensor(10.0, 20.0)

    def test_two_modes_raises(self):
        with pytest.raises(ValueError, match="Only one"):
            RectangularSensor(
                10.0, 20.0, attitude_law=FixedAttitudeLaw.nadir(), body_vector=[0, 0, 1]
            )

    def test_independent_mode_stored(self):
        law = FixedAttitudeLaw.nadir()
        s = RectangularSensor(10.0, 20.0, attitude_law=law)
        assert s._mode == "independent"
        assert s._attitude_law is law

    def test_body_vector_mode_stored(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        assert s._mode == "body"

    def test_body_euler_mode_stored(self):
        s = RectangularSensor(10.0, 20.0, body_euler_deg=(0, 0, 0))
        assert s._mode == "body"

    def test_invalid_attitude_law_type_raises(self):
        with pytest.raises(TypeError, match="AbstractAttitudeLaw"):
            RectangularSensor(10.0, 20.0, attitude_law="nadir")


class TestRectangularSensorAngles:
    def test_theta1_stored_as_radians(self):
        s = RectangularSensor(30.0, 45.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.theta1_rad, np.radians(30.0))
        np.testing.assert_allclose(s.theta2_rad, np.radians(45.0))

    def test_theta_deg_properties(self):
        s = RectangularSensor(30.0, 45.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.theta1_deg, 30.0)
        np.testing.assert_allclose(s.theta2_deg, 45.0)

    def test_90_deg_accepted(self):
        s = RectangularSensor(90.0, 90.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.theta1_rad, np.pi / 2)

    def test_zero_theta1_raises(self):
        with pytest.raises(ValueError, match="theta1_deg"):
            RectangularSensor(0.0, 20.0, body_vector=[0, 0, 1])

    def test_negative_theta2_raises(self):
        with pytest.raises(ValueError, match="theta2_deg"):
            RectangularSensor(10.0, -5.0, body_vector=[0, 0, 1])

    def test_above_90_raises(self):
        with pytest.raises(ValueError, match="theta1_deg"):
            RectangularSensor(91.0, 20.0, body_vector=[0, 0, 1])


# ===========================================================================
# RectangularSensor — body-vector mode
# ===========================================================================


class TestRectangularSensorBodyVector:
    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            RectangularSensor(10.0, 20.0, body_vector=[0, 0, 0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            RectangularSensor(10.0, 20.0, body_vector=[1, 0])

    def test_pointing_raises_before_attach(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        sc = _sc()
        r, v, t = _orbit_state(sc)
        with pytest.raises(RuntimeError, match="add_sensor"):
            s.pointing_eci(r, v, t)

    def test_body_frame_is_orthonormal(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        F = s._body_frame
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-12)

    def test_body_frame_boresight_column_matches(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 5])
        np.testing.assert_allclose(s._body_frame[:, 2], [0, 0, 1], atol=1e-12)


# ===========================================================================
# RectangularSensor — body-euler mode
# ===========================================================================


class TestRectangularSensorBodyEuler:
    def test_identity_euler_boresight_is_body_z(self):
        s = RectangularSensor(10.0, 20.0, body_euler_deg=(0, 0, 0))
        np.testing.assert_allclose(s._body_frame[:, 2], [0, 0, 1], atol=1e-12)

    def test_pitch_90_boresight_is_minus_body_x(self):
        s = RectangularSensor(10.0, 20.0, body_euler_deg=(0, 90, 0))
        np.testing.assert_allclose(s._body_frame[:, 2], [-1, 0, 0], atol=1e-12)

    def test_roll_rotates_perp_axes(self):
        s0 = RectangularSensor(10.0, 20.0, body_euler_deg=(0, 0, 0))
        s1 = RectangularSensor(10.0, 20.0, body_euler_deg=(0, 0, 90))
        # Full frames should differ — roll moves both perp axes and boresight
        assert not np.allclose(s0._body_frame, s1._body_frame)
        # Both frames must still be orthonormal
        np.testing.assert_allclose(
            s0._body_frame @ s0._body_frame.T, np.eye(3), atol=1e-12
        )
        np.testing.assert_allclose(
            s1._body_frame @ s1._body_frame.T, np.eye(3), atol=1e-12
        )

    def test_frame_is_orthonormal(self):
        s = RectangularSensor(10.0, 20.0, body_euler_deg=(30, 45, 60))
        F = s._body_frame
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-12)


# ===========================================================================
# RectangularSensor — independent mode
# ===========================================================================


class TestRectangularSensorIndependent:
    def test_pointing_eci_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s = RectangularSensor(10.0, 20.0, attitude_law=law)
        sc = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_eci(r, v, t),
            law.pointing_eci(r, v, t),
            atol=1e-12,
        )

    def test_no_spacecraft_needed(self):
        s = RectangularSensor(10.0, 20.0, attitude_law=FixedAttitudeLaw.nadir())
        assert s.spacecraft is None
        sc = _sc()
        r, v, t = _orbit_state(sc)
        result = s.pointing_eci(r, v, t)
        assert result.shape == (3,)


# ===========================================================================
# RectangularSensor — sensor frame
# ===========================================================================


class TestRectangularSensorFrame:
    def _setup(self, **kwargs):
        sc = _sc()
        s = RectangularSensor(10.0, 20.0, **kwargs)
        sc.add_sensor(s)
        r, v, t = _orbit_state(sc)
        return s, r, v, t

    def test_body_z_boresight_points_nadir(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_sensor_frame_eci_orthonormal(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        F = s.sensor_frame_eci(r, v, t)
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-10)

    def test_sensor_frame_lvlh_orthonormal(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        F = s.sensor_frame_lvlh(r, v, t)
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-10)

    def test_sensor_frame_boresight_matches_pointing(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        frame = s.sensor_frame_lvlh(r, v, t)
        pointing = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(frame[:, 2], pointing, atol=1e-10)

    def test_independent_sensor_frame_orthonormal(self):
        s, r, v, t = self._setup(attitude_law=FixedAttitudeLaw.nadir())
        F = s.sensor_frame_lvlh(r, v, t)
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-10)

    def test_euler_frame_roll_affects_frame(self):
        s0, r, v, t = self._setup(body_euler_deg=(0, 0, 0))
        s1, _, _, _ = self._setup(body_euler_deg=(0, 0, 45))
        F0 = s0.sensor_frame_lvlh(r, v, t)
        F1 = s1.sensor_frame_lvlh(r, v, t)
        # Full frames should differ
        assert not np.allclose(F0, F1)
        # Both should be orthonormal
        np.testing.assert_allclose(F0 @ F0.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(F1 @ F1.T, np.eye(3), atol=1e-10)


# ===========================================================================
# RectangularSensor — fov_spec
# ===========================================================================


class TestRectangularSensorFovSpec:
    def _setup(self, **kwargs):
        sc = _sc()
        s = RectangularSensor(10.0, 20.0, **kwargs)
        sc.add_sensor(s)
        r, v, t = _orbit_state(sc)
        return s, r, v, t

    def test_fov_spec_has_rectangular_type(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        assert spec["fov_type"] == "rectangular"

    def test_fov_spec_keys(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        for key in (
            "fov_type",
            "pointing_lvlh",
            "u1_lvlh",
            "u2_lvlh",
            "tan_theta1",
            "tan_theta2",
        ):
            assert key in spec

    def test_fov_spec_tan_values(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        np.testing.assert_allclose(spec["tan_theta1"], np.tan(np.radians(10.0)))
        np.testing.assert_allclose(spec["tan_theta2"], np.tan(np.radians(20.0)))

    def test_fov_spec_pointing_is_unit(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        np.testing.assert_allclose(
            np.linalg.norm(spec["pointing_lvlh"]), 1.0, atol=1e-12
        )

    def test_fov_spec_perp_axes_orthogonal_to_boresight(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        p = spec["pointing_lvlh"]
        np.testing.assert_allclose(np.dot(spec["u1_lvlh"], p), 0, atol=1e-12)
        np.testing.assert_allclose(np.dot(spec["u2_lvlh"], p), 0, atol=1e-12)


# ===========================================================================
# RectangularSensor — hierarchy
# ===========================================================================


class TestRectangularSensorHierarchy:
    def test_rectangular_is_subclass(self):
        assert issubclass(RectangularSensor, AbstractSensor)

    def test_rectangular_instance_passes_isinstance(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        assert isinstance(s, AbstractSensor)

    def test_spacecraft_accepts_rectangular(self):
        sc = _sc()
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)
        assert s.spacecraft is sc

    def test_repr_body_mode(self):
        s = RectangularSensor(10.0, 20.0, body_vector=[0, 0, 1])
        r = repr(s)
        assert "RectangularSensor" in r
        assert "mode='body'" in r

    def test_repr_independent_mode(self):
        s = RectangularSensor(10.0, 20.0, attitude_law=FixedAttitudeLaw.nadir())
        r = repr(s)
        assert "RectangularSensor" in r
        assert "attitude_law" in r


# ===========================================================================
# ConicSensor — fov_spec
# ===========================================================================


class TestConicSensorFovSpec:
    def _setup(self, **kwargs):
        sc = _sc()
        s = ConicSensor(30.0, **kwargs)
        sc.add_sensor(s)
        r, v, t = _orbit_state(sc)
        return s, r, v, t

    def test_fov_spec_has_conic_type(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        assert spec["fov_type"] == "conic"

    def test_fov_spec_keys(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        assert "pointing_lvlh" in spec
        assert "cos_half_angle" in spec

    def test_fov_spec_cos_value(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        np.testing.assert_allclose(spec["cos_half_angle"], np.cos(np.radians(30.0)))

    def test_fov_spec_pointing_is_unit(self):
        s, r, v, t = self._setup(body_vector=[0, 0, 1])
        spec = s.fov_spec(r, v, t)
        np.testing.assert_allclose(
            np.linalg.norm(spec["pointing_lvlh"]), 1.0, atol=1e-12
        )
