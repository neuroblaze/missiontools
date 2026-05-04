"""
missiontools.sensor.sensor_law
==============================
Sensor classes for instruments attached to a spacecraft.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from ..orbit.frames import eci_to_lvlh, eci_to_ecef


def _euler_zyx_to_boresight(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> npt.NDArray[np.floating]:
    """Sensor boresight in spacecraft body frame from ZYX Euler angles.

    The ZYX intrinsic rotation sequence (yaw → pitch → roll) defines the
    rotation from the spacecraft body frame to the sensor frame:
    ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``.

    The sensor boresight (sensor frame z-axis = ``[0, 0, 1]``) expressed
    in the spacecraft body frame is ``R.T @ [0, 0, 1]``.

    Parameters
    ----------
    yaw_deg : float
        Yaw angle (deg), rotation about body-Z.
    pitch_deg : float
        Pitch angle (deg), rotation about new Y after yaw.
    roll_deg : float
        Roll angle (deg), rotation about new X after pitch.

    Returns
    -------
    npt.NDArray[np.floating], shape (3,)
        Unit boresight vector in spacecraft body frame.
    """
    R = _euler_zyx_to_sensor_frame(yaw_deg, pitch_deg, roll_deg)
    return R[:, 2]


def _euler_zyx_to_sensor_frame(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> npt.NDArray[np.floating]:
    """Full sensor frame in spacecraft body coordinates from ZYX Euler angles.

    The ZYX intrinsic rotation sequence (yaw → pitch → roll) defines the
    rotation from the spacecraft body frame to the sensor frame:
    ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``.

    The returned matrix columns are the sensor-x, sensor-y, and sensor-z
    (boresight) axes expressed in the spacecraft body frame.

    Parameters
    ----------
    yaw_deg : float
        Yaw angle (deg), rotation about body-Z.
    pitch_deg : float
        Pitch angle (deg), rotation about new Y after yaw.
    roll_deg : float
        Roll angle (deg), rotation about new X after pitch.

    Returns
    -------
    npt.NDArray[np.floating], shape (3, 3)
        Columns are sensor-x, sensor-y, sensor-z in body frame.
        Equivalent to ``R.T`` where ``R`` is the body-to-sensor rotation.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R = Rx @ Ry @ Rz
    return R.T


def _orthonormal_frame(boresight: npt.NDArray) -> npt.NDArray:
    """Build an orthonormal (3, 3) frame from a boresight via Gram-Schmidt.

    Columns are [u1, u2, boresight] where u1, u2 are perpendicular to
    *boresight* and to each other.  The reference direction is [0, 0, 1];
    if the boresight is nearly parallel to that, [1, 0, 0] is used instead.
    """
    b = boresight / np.linalg.norm(boresight)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(b, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u1 = ref - np.dot(ref, b) * b
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(b, u1)
    u2 /= np.linalg.norm(u2)
    return np.column_stack([u1, u2, b])


class AbstractSensor(ABC):
    """Abstract base class for instruments attached to a spacecraft.

    Subclasses must implement :meth:`pointing_eci`, :meth:`fov_spec`,
    and :meth:`__repr__`.
    The concrete :meth:`pointing_lvlh` and :meth:`pointing_ecef` methods are
    provided here and delegate to :meth:`pointing_eci` plus a frame transform.
    """

    def __init__(self, *, condition=None) -> None:
        self._spacecraft = None  # set by Spacecraft.add_sensor
        self._condition = None
        if condition is not None:
            self.condition = condition

    @property
    def spacecraft(self):
        """Host spacecraft, or ``None`` if not yet attached."""
        return self._spacecraft

    @property
    def condition(self):
        """Optional :class:`~missiontools.AbstractCondition` controlling when this
        sensor is active, or ``None`` (always active)."""
        return self._condition

    @condition.setter
    def condition(self, value):
        from ..condition import AbstractCondition

        if value is not None and not isinstance(value, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition instance or None, "
                f"got {type(value).__name__!r}"
            )
        self._condition = value

    @abstractmethod
    def pointing_eci(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECI frame."""

    @abstractmethod
    def fov_spec(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> dict[str, Any]:
        """Return FOV parameters frozen at the given orbital state.

        Returns a dict with at minimum:

        - ``'fov_type'``: ``'conic'`` or ``'rectangular'``
        - ``'pointing_lvlh'``: ``(3,)`` boresight unit vector in LVLH

        Conic specs also include ``'cos_half_angle'``.
        Rectangular specs also include ``'u1_lvlh'``, ``'u2_lvlh'``,
        ``'tan_theta1'``, and ``'tan_theta2'``.
        """

    @abstractmethod
    def __repr__(self) -> str: ...

    def pointing_lvlh(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the LVLH frame.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in LVLH, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        r = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        r_2d = np.atleast_2d(r)
        v_2d = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        eci = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))
        result = eci_to_lvlh(eci, r_2d, v_2d)
        return result[0] if scalar else result

    def pointing_ecef(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECEF frame.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in ECEF, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        r = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        t_arr = np.atleast_1d(np.asarray(t, dtype="datetime64[us]"))
        eci = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))
        result = eci_to_ecef(eci, t_arr)
        return result[0] if scalar else result


class ConicSensor(AbstractSensor):
    """An instrument attached to a spacecraft with a cone-shaped field of view.

    Prefer the keyword arguments to select the pointing mode (see below).
    The constructor is public and may be called directly when needed.

    Parameters
    ----------
    half_angle_deg : float
        Half-angle of the sensor's conical field of view (degrees).
        Must satisfy ``0 < half_angle_deg <= 90``.
    attitude_law : AbstractAttitudeLaw, optional
        Independent :class:`~missiontools.AbstractAttitudeLaw` for this sensor,
        decoupled from the host spacecraft's attitude.  Mutually exclusive
        with ``body_vector`` and ``body_euler_deg``.
    body_vector : array_like, shape (3,), optional
        Boresight direction expressed in the **spacecraft body frame**.
        Normalised on input.  Mutually exclusive with ``attitude_law`` and
        ``body_euler_deg``.
    body_euler_deg : (yaw, pitch, roll) tuple of float, optional
        ZYX intrinsic Euler angles (degrees) defining the sensor frame
        relative to the spacecraft body frame.  The boresight is the
        sensor frame's z-axis expressed in body-frame coordinates.
        Mutually exclusive with ``attitude_law`` and ``body_vector``.

    Notes
    -----
    Body-mounted sensors (``body_vector`` or ``body_euler_deg``) require the
    sensor to be attached to a spacecraft via
    :meth:`~missiontools.Spacecraft.add_sensor` before their pointing methods
    can be called.

    Examples
    --------
    Nadir-pointing sensor, 10° half-angle::

        from missiontools import ConicSensor, FixedAttitudeLaw
        sensor = ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir())

    Sensor body-mounted along spacecraft body-z (boresight = nadir for a
    nadir spacecraft), 5° half-angle::

        sensor = ConicSensor(5.0, body_vector=[0, 0, 1])

    Sensor tilted 30° in pitch from body-z::

        sensor = ConicSensor(15.0, body_euler_deg=(0, 30, 0))
    """

    def __init__(
        self,
        half_angle_deg: float,
        *,
        attitude_law=None,
        body_vector: npt.ArrayLike | None = None,
        body_euler_deg: tuple[float, float, float] | None = None,
        condition=None,
    ):
        super().__init__(condition=condition)

        # --- validate half-angle -------------------------------------------
        half_angle_deg = float(half_angle_deg)
        if not (0.0 < half_angle_deg <= 90.0):
            raise ValueError(f"half_angle_deg must be in (0, 90], got {half_angle_deg}")
        self._half_angle_rad: float = np.radians(half_angle_deg)

        # --- validate mode (exactly one) ------------------------------------
        n_modes = sum(
            x is not None for x in (attitude_law, body_vector, body_euler_deg)
        )
        if n_modes == 0:
            raise ValueError(
                "Exactly one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' must be provided."
            )
        if n_modes > 1:
            raise ValueError(
                "Only one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' may be provided."
            )

        # --- store mode-specific state --------------------------------------
        if attitude_law is not None:
            from ..attitude import AbstractAttitudeLaw

            if not isinstance(attitude_law, AbstractAttitudeLaw):
                raise TypeError(
                    f"attitude_law must be an AbstractAttitudeLaw instance, "
                    f"got {type(attitude_law).__name__!r}"
                )
            self._mode = "independent"
            self._attitude_law = attitude_law
            self._body_vector = None

        elif body_vector is not None:
            vec = np.asarray(body_vector, dtype=np.float64)
            if vec.shape != (3,):
                raise ValueError(f"body_vector must have shape (3,), got {vec.shape}")
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                raise ValueError("body_vector must not be the zero vector")
            self._mode = "body"
            self._attitude_law = None
            self._body_vector = vec / norm

        else:  # body_euler_deg
            yaw, pitch, roll = (float(a) for a in body_euler_deg)
            boresight = _euler_zyx_to_boresight(yaw, pitch, roll)
            self._mode = "body"
            self._attitude_law = None
            self._body_vector = boresight / np.linalg.norm(boresight)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def half_angle_rad(self) -> float:
        """FOV cone half-angle in radians."""
        return self._half_angle_rad

    @property
    def half_angle_deg(self) -> float:
        """FOV cone half-angle in degrees."""
        return float(np.degrees(self._half_angle_rad))

    # ------------------------------------------------------------------
    # Pointing
    # ------------------------------------------------------------------

    def pointing_eci(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECI frame.

        For ``body_vector`` / ``body_euler_deg`` sensors the host spacecraft's
        :attr:`~missiontools.Spacecraft.attitude_law` is used to transform
        the body-frame boresight to ECI.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in ECI, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.

        Raises
        ------
        RuntimeError
            If the sensor is in body mode and has not been attached to a
            spacecraft via :meth:`~missiontools.Spacecraft.add_sensor`.
        """
        if self._mode == "independent":
            return self._attitude_law.pointing_eci(r_eci, v_eci, t)

        # body mode
        if self._spacecraft is None:
            raise RuntimeError(
                "Sensor must be attached to a Spacecraft via add_sensor() "
                "before pointing methods can be called in body mode."
            )
        return self._spacecraft.attitude_law.rotate_from_body(
            self._body_vector,
            r_eci,
            v_eci,
            t,
        )

    def __repr__(self) -> str:
        cond_part = (
            f", condition={self._condition!r}" if self._condition is not None else ""
        )
        if self._mode == "independent":
            return (
                f"ConicSensor(half_angle_deg={self.half_angle_deg:.3f}, "
                f"attitude_law={self._attitude_law!r}{cond_part})"
            )
        return (
            f"ConicSensor(half_angle_deg={self.half_angle_deg:.3f}, "
            f"mode='body', body_vector={self._body_vector.tolist()}{cond_part})"
        )

    def fov_spec(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> dict[str, Any]:
        pointing_lvlh = self.pointing_lvlh(r_eci, v_eci, t)
        if pointing_lvlh.ndim == 2:
            pointing_lvlh = pointing_lvlh[0]
        return {
            "fov_type": "conic",
            "pointing_lvlh": pointing_lvlh,
            "cos_half_angle": float(np.cos(self._half_angle_rad)),
        }


class RectangularSensor(AbstractSensor):
    """An instrument with a rectangular (pyramidal) field of view.

    The FOV is defined by two half-angles, *theta1* and *theta2*, measured
    in two orthogonal planes of the sensor frame.  Together they describe
    a rectangular pyramid whose apex is at the sensor.

    Parameters
    ----------
    theta1_deg : float
        Half-angle in the sensor x-z plane (degrees).
        Must satisfy ``0 < theta1_deg <= 90``.
    theta2_deg : float
        Half-angle in the sensor y-z plane (degrees).
        Must satisfy ``0 < theta2_deg <= 90``.
    attitude_law : AbstractAttitudeLaw, optional
        Independent :class:`~missiontools.AbstractAttitudeLaw` for this sensor,
        decoupled from the host spacecraft's attitude.  Mutually exclusive
        with ``body_vector`` and ``body_euler_deg``.
    body_vector : array_like, shape (3,), optional
        Boresight direction expressed in the **spacecraft body frame**.
        Normalised on input.  Mutually exclusive with ``attitude_law`` and
        ``body_euler_deg``.
    body_euler_deg : (yaw, pitch, roll) tuple of float, optional
        ZYX intrinsic Euler angles (degrees) defining the sensor frame
        relative to the spacecraft body frame.  The boresight is the
        sensor frame's z-axis; the x/y axes define the FOV orientation
        about the boresight (roll matters for rectangular sensors).
        Mutually exclusive with ``attitude_law`` and ``body_vector``.

    Notes
    -----
    Body-mounted sensors require attachment to a spacecraft via
    :meth:`~missiontools.Spacecraft.add_sensor` before their pointing methods
    can be called.

    For ``body_vector`` mode the perpendicular axes are derived via
    Gram-Schmidt orthogonalisation against the body-z direction (falling
    back to body-x if the boresight is nearly parallel to body-z).

    Examples
    --------
    Nadir-pointing rectangular sensor, 10° × 20°::

        from missiontools import RectangularSensor, FixedAttitudeLaw
        sensor = RectangularSensor(10.0, 20.0,
                                   attitude_law=FixedAttitudeLaw.nadir())

    Body-mounted with Euler angles (roll rotates the FOV about boresight)::

        sensor = RectangularSensor(5.0, 10.0, body_euler_deg=(0, 30, 45))
    """

    def __init__(
        self,
        theta1_deg: float,
        theta2_deg: float,
        *,
        attitude_law=None,
        body_vector: npt.ArrayLike | None = None,
        body_euler_deg: tuple[float, float, float] | None = None,
        condition=None,
    ):
        super().__init__(condition=condition)

        theta1_deg = float(theta1_deg)
        theta2_deg = float(theta2_deg)
        if not (0.0 < theta1_deg <= 90.0):
            raise ValueError(f"theta1_deg must be in (0, 90], got {theta1_deg}")
        if not (0.0 < theta2_deg <= 90.0):
            raise ValueError(f"theta2_deg must be in (0, 90], got {theta2_deg}")
        self._theta1_rad: float = np.radians(theta1_deg)
        self._theta2_rad: float = np.radians(theta2_deg)

        n_modes = sum(
            x is not None for x in (attitude_law, body_vector, body_euler_deg)
        )
        if n_modes == 0:
            raise ValueError(
                "Exactly one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' must be provided."
            )
        if n_modes > 1:
            raise ValueError(
                "Only one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' may be provided."
            )

        if attitude_law is not None:
            from ..attitude import AbstractAttitudeLaw

            if not isinstance(attitude_law, AbstractAttitudeLaw):
                raise TypeError(
                    f"attitude_law must be an AbstractAttitudeLaw instance, "
                    f"got {type(attitude_law).__name__!r}"
                )
            self._mode = "independent"
            self._attitude_law = attitude_law
            self._body_frame: npt.NDArray | None = None

        elif body_vector is not None:
            vec = np.asarray(body_vector, dtype=np.float64)
            if vec.shape != (3,):
                raise ValueError(f"body_vector must have shape (3,), got {vec.shape}")
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                raise ValueError("body_vector must not be the zero vector")
            boresight = vec / norm
            self._mode = "body"
            self._attitude_law = None
            self._body_frame = _orthonormal_frame(boresight)

        else:  # body_euler_deg
            yaw, pitch, roll = (float(a) for a in body_euler_deg)
            frame = _euler_zyx_to_sensor_frame(yaw, pitch, roll)
            self._mode = "body"
            self._attitude_law = None
            self._body_frame = frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def theta1_rad(self) -> float:
        """FOV half-angle in the sensor x-z plane (radians)."""
        return self._theta1_rad

    @property
    def theta1_deg(self) -> float:
        """FOV half-angle in the sensor x-z plane (degrees)."""
        return float(np.degrees(self._theta1_rad))

    @property
    def theta2_rad(self) -> float:
        """FOV half-angle in the sensor y-z plane (radians)."""
        return self._theta2_rad

    @property
    def theta2_deg(self) -> float:
        """FOV half-angle in the sensor y-z plane (degrees)."""
        return float(np.degrees(self._theta2_rad))

    # ------------------------------------------------------------------
    # Pointing
    # ------------------------------------------------------------------

    def _require_body(self) -> None:
        if self._spacecraft is None:
            raise RuntimeError(
                "Sensor must be attached to a Spacecraft via add_sensor() "
                "before pointing methods can be called in body mode."
            )

    def sensor_frame_eci(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Sensor frame axes in ECI as columns of a (3, 3) matrix.

        Returns ``[u1 | u2 | boresight]`` where each column is a unit
        vector in ECI.
        """
        if self._mode == "independent":
            law = self._attitude_law
            u1 = law.rotate_from_body(np.array([1.0, 0.0, 0.0]), r_eci, v_eci, t)
            u2 = law.rotate_from_body(np.array([0.0, 1.0, 0.0]), r_eci, v_eci, t)
            bz = law.rotate_from_body(np.array([0.0, 0.0, 1.0]), r_eci, v_eci, t)
            if u1.ndim == 2:
                u1, u2, bz = u1[0], u2[0], bz[0]
            return np.column_stack([u1, u2, bz])

        self._require_body()
        law = self._spacecraft.attitude_law
        frame_body = self._body_frame
        u1 = law.rotate_from_body(frame_body[:, 0], r_eci, v_eci, t)
        u2 = law.rotate_from_body(frame_body[:, 1], r_eci, v_eci, t)
        bz = law.rotate_from_body(frame_body[:, 2], r_eci, v_eci, t)
        if u1.ndim == 2:
            u1, u2, bz = u1[0], u2[0], bz[0]
        return np.column_stack([u1, u2, bz])

    def sensor_frame_lvlh(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Sensor frame axes in LVLH as columns of a (3, 3) matrix."""
        frame_eci = self.sensor_frame_eci(r_eci, v_eci, t)
        r = np.asarray(r_eci, dtype=np.float64)
        r_2d = np.atleast_2d(r)
        v_2d = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        return eci_to_lvlh(frame_eci.T, r_2d, v_2d).T

    def pointing_eci(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        if self._mode == "independent":
            return self._attitude_law.pointing_eci(r_eci, v_eci, t)

        self._require_body()
        return self._spacecraft.attitude_law.rotate_from_body(
            self._body_frame[:, 2],
            r_eci,
            v_eci,
            t,
        )

    def fov_spec(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> dict[str, Any]:
        frame = self.sensor_frame_lvlh(r_eci, v_eci, t)
        return {
            "fov_type": "rectangular",
            "pointing_lvlh": frame[:, 2].copy(),
            "u1_lvlh": frame[:, 0].copy(),
            "u2_lvlh": frame[:, 1].copy(),
            "tan_theta1": float(np.tan(self._theta1_rad)),
            "tan_theta2": float(np.tan(self._theta2_rad)),
        }

    def __repr__(self) -> str:
        cond_part = (
            f", condition={self._condition!r}" if self._condition is not None else ""
        )
        if self._mode == "independent":
            return (
                f"RectangularSensor(theta1_deg={self.theta1_deg:.3f}, "
                f"theta2_deg={self.theta2_deg:.3f}, "
                f"attitude_law={self._attitude_law!r}{cond_part})"
            )
        return (
            f"RectangularSensor(theta1_deg={self.theta1_deg:.3f}, "
            f"theta2_deg={self.theta2_deg:.3f}, mode='body'{cond_part})"
        )
