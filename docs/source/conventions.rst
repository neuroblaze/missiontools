Conventions
===========

Units
-----
All physical quantities use **SI base units** (m, kg, s, A, K, …) unless
explicitly stated otherwise in a function's docstring.  Notable exceptions
that are called out everywhere they appear:

* Angles — **radians** throughout the library; degrees appear only in
  user-facing constructors (e.g. ``GroundStation(lat, lon)`` accepts degrees).
* Orbital elements — semi-major axis in metres, angles in radians.
* Antenna gain — dBi; power levels — dBW.

Coordinate frames
-----------------
* **ECI** (Earth-Centred Inertial, J2000/GCRS): the default frame for
  positions and velocities.  The x-axis points toward the vernal equinox,
  z toward the celestial north pole.
* **ECEF** (Earth-Centred Earth-Fixed, WGS84): rotates with the Earth.
  Used for ground-station positions.
* **LVLH** (Local Vertical Local Horizontal): body-frame reference aligned
  with the orbital plane.  x = along-track, y = cross-track, z = nadir.

Array conventions
-----------------
* Position / velocity arrays are shaped ``(N, 3)`` for a time series or
  ``(3,)`` for a single epoch.  Scalar detection uses ``.ndim == 1``.
* Times are ``numpy.datetime64[us]`` throughout.

Attitude
--------
* Quaternions are stored as ``[w, x, y, z]`` (scalar-first).
* The spacecraft **boresight** is the body-z axis.
* **Nadir** mode: body-z points nadir (−R̂), body-x along-track (Ŝ).
