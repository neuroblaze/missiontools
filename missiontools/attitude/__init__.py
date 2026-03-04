"""
missiontools.attitude
=====================
Spacecraft and sensor pointing laws.

The primary class is :class:`AttitudeLaw`, which stores a full 3-DOF body
orientation as a unit quaternion ``[w, x, y, z]`` and supports several
pointing modes:

- **Nadir** — body-z toward the Earth centre (default for all
  :class:`~missiontools.Spacecraft` instances).
- **Fixed** — constant body orientation in a chosen reference frame
  (LVLH, ECI, or ECEF).  The boresight direction and an optional roll
  angle are specified at construction.
- **Track** — boresight pointing toward a target
  :class:`~missiontools.Spacecraft` at every timestep.

All pointing methods (:meth:`~AttitudeLaw.pointing_eci`,
:meth:`~AttitudeLaw.pointing_lvlh`, :meth:`~AttitudeLaw.pointing_ecef`)
return the **body-z** unit vector expressed in the requested frame.

Optional yaw steering can be enabled via
:meth:`~AttitudeLaw.yaw_steering` to maximise solar power generation by
rotating the spacecraft about the boresight axis at each timestep.

Planned functionality
---------------------
- Environmental disturbance torques
- Actuator sizing (reaction wheels, magnetorquers, thrusters)
- Pointing budget and error analysis
- Sensor modelling (star tracker, sun sensor, magnetometer)
"""

from .attitude_law import AttitudeLaw

__all__ = ['AttitudeLaw']
