import numpy as np
import numpy.typing as npt
from datetime import datetime
from .constants import EARTH_MU, EARTH_J2, EARTH_MEAN_RADIUS, EARTH_SEMI_MAJOR_AXIS

# Mean solar motion: 2π rad per Julian year (365.25 days)
_N_SUN = 2.0 * np.pi / (365.25 * 86400.0)   # rad/s

# J2000.0 epoch (same as frames.py)
_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')


def propagate_analytical(t: list[datetime] | npt.NDArray[np.datetime64],
                         epoch: datetime | np.datetime64,
                         a: float | np.floating,
                         e: float | np.floating,
                         i: float | np.floating,
                         arg_p: float | np.floating,
                         raan: float | np.floating,
                         ma: float | np.floating,
                         type: str = "twobody",
                         central_body_mu: float | np.floating = EARTH_MU,
                         central_body_j2: float | np.floating = EARTH_J2,
                         central_body_radius: float | np.floating = EARTH_MEAN_RADIUS) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Propagate a Keplerian orbit analytically and return ECI state vectors.

    Advances the mean anomaly from ``epoch`` to each time in ``t`` using the
    selected propagation model, solves Kepler's equation for the eccentric
    anomaly, and transforms the result into ECI Cartesian position and
    velocity vectors.

    Parameters
    ----------
    t : list[datetime] | npt.NDArray[np.datetime64]
        Times at which to evaluate the state vector.
    epoch : datetime | np.datetime64
        Epoch of the supplied orbital elements (i.e. the time at which ``ma``
        is defined).
    a : float | np.floating
        Semi-major axis (m).
    e : float | np.floating
        Eccentricity (dimensionless, 0 <= e < 1 for elliptical orbits).
    i : float | np.floating
        Inclination (rad).
    arg_p : float | np.floating
        Argument of perigee (rad).
    raan : float | np.floating
        Right ascension of the ascending node (rad).
    ma : float | np.floating
        Mean anomaly at epoch (rad).
    type : str, optional
        Propagation model to use. ``"twobody"`` (default) uses unperturbed
        Keplerian motion. ``"j2"`` incorporates mean J2 perturbations
    central_body_mu : float | np.floating, optional
        Standard gravitational parameter of the central body (m³/s²).
        Defaults to ``EARTH_MU``.
    central_body_j2 : float | np.floating, optional
        J2 perturbation parameter of the central body (m⁵/s²), defined as
        μ × J₂ × R². Only used when ``type`` includes J2 perturbations.
        Defaults to ``EARTH_J2``.

    Returns
    -------
    r : npt.NDArray[np.floating]
        Position vectors in the ECI frame, shape ``(N, 3)`` (m).
    v : npt.NDArray[np.floating]
        Velocity vectors in the ECI frame, shape ``(N, 3)`` (m/s).
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must satisfy 0 <= e < 1 for elliptical orbits, got e={e}")
    if not (0 <= i <= np.pi):
        raise ValueError(f"Inclination must be in [0, π], got i={i}")
    for name, val in (("arg_p", arg_p), ("raan", raan), ("ma", ma)):
        if not (0 <= val < 2*np.pi):
            raise ValueError(f"Angle '{name}' must be in [0, 2π), got {name}={val}")
    if central_body_mu <= 0:
        raise ValueError(f"central_body_mu must be positive, got central_body_mu={central_body_mu}")
    
    # cast datetimes to microsecond precision
    t = np.asarray(t, dtype='datetime64[us]')

    # get time offset from epoch in floating-point seconds
    t_e = (t - np.datetime64(epoch).astype('datetime64[us]')).astype(np.float64)*1e-6

    # compute mean motion
    mean_motion = np.sqrt(central_body_mu/a**3)

    # compute semi-latus rectum
    p = a*(1-e**2)

    if type == "j2":
        # compute secular J2 terms
        # central_body_j2 = mu * J2_dim * R^2, so dividing by mu gives J2_dim * R^2 (m^2),
        # the quantity that appears in the Vallado secular rate formulas.
        j2_r2 = central_body_j2 / central_body_mu

        # Vallado eq. 9-37
        raan_dot = (-3.0*mean_motion*j2_r2) / \
                   (2.0*p**2) * np.cos(i)
        raan = raan + raan_dot*t_e

        # Vallado eq. 9-39
        arg_p_dot = (3.0*mean_motion*j2_r2) / \
                    (4.0*p**2) * (4.0-5.0*np.sin(i)**2)
        arg_p = arg_p + arg_p_dot*t_e

        # Vallado eq. 9-41
        ma_dot = (3.0*mean_motion*j2_r2*np.sqrt(1-e**2)) / \
                 (4.0*p**2) * (2.0 - 3.0*np.sin(i)**2)
        ma_t = (ma + (mean_motion + ma_dot)*t_e) % (2*np.pi)
    else:
        # two body case - raan, arg_p, ma remain at initial values

        # compute mean anomaly
        ma_t = (ma + mean_motion*t_e) % (2*np.pi)

    # compute eccentric and true anomaly
    if (e == 0.0):
        # optimization for circular orbits
        ta = ma_t
        E = ma_t
    else:
        # solve for the eccentric anomaly
        # 5 newton iterations sufficient for double FP
        E = ma_t
        for n in range(5):
            E = E - (E-e*np.sin(E) - ma_t)/(1-e*np.cos(E))
        
        # compute the true anomaly
        ta = 2.0*np.arctan2(np.sqrt(1+e)*np.sin(E/2),
                            np.sqrt(1-e)*np.cos(E/2))
    
    # compute distance to central body
    if (e == 0.0):
        rc = a
    else:
        rc = a*(1-e*np.cos(E))

    # get position in orbit plane
    or_t = rc*np.array([np.cos(ta),
                        np.sin(ta),
                        np.repeat(0.0, ta.shape[0])])
    
    # get velocity in orbit plane
    ov_t = (np.sqrt(central_body_mu*a)/rc)*np.array([-1*np.sin(E),
                                                     np.sqrt(1-e**2)*np.cos(E),
                                                     np.repeat(0.0, E.shape[0])])

    if type == "twobody":
        # since rotations are static over time for two body,
        # fastest way to implement for large number of times
        # is construct a single combined rotation matrix

        # construct rotation matrices
        # Rz(-raan) - rotate about Z by -RAAN
        rot_raan = np.array([[ np.cos(raan),  np.sin(raan),  0.0],
                            [-np.sin(raan),  np.cos(raan),  0.0],
                            [ 0.0,           0.0,           1.0]])

        # Rx(-i) - rotate about X by -inclination
        rot_i = np.array([[1.0,  0.0,          0.0        ],
                        [0.0,  np.cos(i),    np.sin(i)  ],
                        [0.0, -np.sin(i),    np.cos(i)  ]])

        # Rz(-arg_p)  — rotate about Z by -argument of perigee
        rot_arg_p = np.array([[ np.cos(arg_p),  np.sin(arg_p),  0.0],
                            [-np.sin(arg_p),  np.cos(arg_p),  0.0],
                            [ 0.0,            0.0,            1.0]])

        # Combined perifocal to ECI
        R = rot_raan @ rot_i @ rot_arg_p

        # generate output vectors
        r = (R @ or_t).T
        v = (R @ ov_t).T
    else:
        # For J2, raan and arg_p are (N,) arrays (secular drift applied above),
        # so each timestep needs its own rotation matrix.
        # Build (N, 3, 3) rotation stacks and compose with einsum.

        z = np.zeros_like(raan)
        o = np.ones_like(raan)

        # Rz(-raan[k]) for each k — shape (N, 3, 3)
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        rot_raan = np.array([[ cos_raan,  sin_raan, z],
                             [-sin_raan,  cos_raan, z],
                             [z,          z,        o]]).transpose(2, 0, 1)

        # Rx(-i) — scalar i, shape (3, 3)
        rot_i = np.array([[1.0,  0.0,         0.0       ],
                          [0.0,  np.cos(i),   np.sin(i) ],
                          [0.0, -np.sin(i),   np.cos(i) ]])

        # Rz(-arg_p[k]) for each k — shape (N, 3, 3)
        cos_arg_p, sin_arg_p = np.cos(arg_p), np.sin(arg_p)
        rot_arg_p = np.array([[ cos_arg_p,  sin_arg_p, z],
                              [-sin_arg_p,  cos_arg_p, z],
                              [z,           z,         o]]).transpose(2, 0, 1)

        # Combined R[k] = rot_raan[k] @ rot_i @ rot_arg_p[k]
        # (3,3) @ (N,3,3) broadcasts to (N,3,3); then (N,3,3) @ (N,3,3) → (N,3,3)
        R = rot_raan @ (rot_i @ rot_arg_p)

        # Apply each R[k] to the corresponding perifocal column: or_t is (3,N)
        r = np.einsum('nij,jn->ni', R, or_t)
        v = np.einsum('nij,jn->ni', R, ov_t)

    # result
    return r, v


def sun_synchronous_inclination(
        a: float | np.floating,
        e: float | np.floating = 0.0,
        central_body_mu: float | np.floating = EARTH_MU,
        central_body_j2: float | np.floating = EARTH_J2,
) -> float:
    """Return the inclination (rad) required for a sun-synchronous orbit.

    A sun-synchronous orbit is one whose RAAN precesses at exactly the mean
    solar rate (+2π rad per Julian year), keeping the orbital plane at a
    roughly constant angle with respect to the Sun. The required inclination
    is derived by setting the J2 secular RAAN drift rate equal to the mean
    solar motion and solving for *i*.

    From Vallado eq. 9-37:

    .. math::

        \\dot{\\Omega} = -\\frac{3}{2} \\frac{n J_2 R^2}{p^2} \\cos i
        \\;=\\; n_{\\odot}

    Solving for inclination::

        \\cos i = -\\frac{2 n_{\\odot} p^2}{3 n J_2 R^2}

    where :math:`p = a(1-e^2)` is the semi-latus rectum,
    :math:`n = \\sqrt{\\mu / a^3}` is the mean motion, and
    :math:`J_2 R^2 = ` ``central_body_j2 / central_body_mu``.

    Parameters
    ----------
    a : float | np.floating
        Semi-major axis (m).
    e : float | np.floating, optional
        Eccentricity (dimensionless, 0 <= e < 1). Defaults to 0 (circular).
    central_body_mu : float | np.floating, optional
        Standard gravitational parameter (m³/s²). Defaults to ``EARTH_MU``.
    central_body_j2 : float | np.floating, optional
        Combined J2 parameter μ × J₂_dim × R² (m⁵/s²). Defaults to
        ``EARTH_J2``.

    Returns
    -------
    float
        Sun-synchronous inclination in radians (will be in (π/2, π) for
        a prograde-retrograde orbit around Earth, typically ~97–100°).

    Raises
    ------
    ValueError
        If ``a`` or ``central_body_mu`` are non-positive, if ``e`` is
        outside [0, 1), or if no sun-synchronous orbit exists for the
        supplied parameters (``|cos i| > 1``).
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if not (0 <= e < 1):
        raise ValueError(
            f"Eccentricity must satisfy 0 <= e < 1, got e={e}"
        )
    if central_body_mu <= 0:
        raise ValueError(
            f"central_body_mu must be positive, got {central_body_mu}"
        )

    n      = np.sqrt(central_body_mu / a**3)           # mean motion (rad/s)
    p      = a * (1.0 - e**2)                          # semi-latus rectum (m)
    j2_r2  = central_body_j2 / central_body_mu         # J₂_dim × R²  (m²)

    cos_i = (-2.0 * _N_SUN * p**2) / (3.0 * n * j2_r2)

    if abs(cos_i) > 1.0:
        raise ValueError(
            f"No sun-synchronous orbit exists for a={a} m, e={e}: "
            f"cos(i) = {cos_i:.4f} is outside [-1, 1]."
        )

    return float(np.arccos(cos_i))


def sun_synchronous_orbit(
        altitude:            float | np.floating,
        local_time_at_node:  str,
        node_type:           str = 'ascending',
        epoch:               datetime | np.datetime64 | None = None,
        central_body_mu:     float | np.floating = EARTH_MU,
        central_body_j2:     float | np.floating = EARTH_J2,
        central_body_radius: float | np.floating = EARTH_SEMI_MAJOR_AXIS,
) -> dict:
    """Return Keplerian elements for a circular sun-synchronous orbit.

    Computes the RAAN such that the specified node type crosses the equator
    at the requested local solar time on the given epoch, and the inclination
    that produces a sun-synchronous RAAN drift rate.

    The returned dict is ready to unpack directly into
    :func:`propagate_analytical` (``propagate_analytical(t, **params)``).

    Parameters
    ----------
    altitude : float | np.floating
        Orbit altitude above the body's equatorial surface (m).
    local_time_at_node : str
        Local solar time at the specified node crossing, formatted as
        ``"HH:MM"`` or ``"HH:MM:SS"`` (24-hour clock).
    node_type : str, optional
        ``'ascending'`` (default) or ``'descending'``. Indicates which node
        crossing the local time refers to.
    epoch : datetime | np.datetime64 | None, optional
        Epoch at which the orbital elements are defined and the node crossing
        occurs. Defaults to J2000.0 (``2000-01-01T12:00:00`` UTC).
    central_body_mu : float | np.floating, optional
        Standard gravitational parameter (m³/s²). Defaults to ``EARTH_MU``.
    central_body_j2 : float | np.floating, optional
        Combined J2 parameter μ × J₂_dim × R² (m⁵/s²). Defaults to
        ``EARTH_J2``.
    central_body_radius : float | np.floating, optional
        Equatorial radius used for altitude→semi-major-axis conversion (m).
        Defaults to ``EARTH_SEMI_MAJOR_AXIS`` (WGS84 equatorial radius).

    Returns
    -------
    dict
        Keplerian parameter dict with keys ``epoch``, ``a``, ``e``,
        ``i``, ``arg_p``, ``raan``, ``ma``, ``central_body_mu``,
        ``central_body_j2``, ``central_body_radius``.  All angles are in
        radians; ``epoch`` is ``datetime64[us]``.

    Raises
    ------
    ValueError
        If ``local_time_at_node`` cannot be parsed, ``node_type`` is not
        ``'ascending'`` or ``'descending'``, ``altitude`` is negative, or
        no sun-synchronous orbit exists for the given parameters.
    """
    # --- epoch ---
    if epoch is None:
        epoch = _J2000_US
    epoch_us = np.asarray(epoch, dtype='datetime64[us]')

    # --- parse local time string ---
    parts = local_time_at_node.strip().split(':')
    if len(parts) == 2:
        hh, mm, ss = parts[0], parts[1], '0'
    elif len(parts) == 3:
        hh, mm, ss = parts[0], parts[1], parts[2]
    else:
        raise ValueError(
            f"local_time_at_node must be 'HH:MM' or 'HH:MM:SS', "
            f"got '{local_time_at_node}'"
        )
    try:
        lsol = int(hh) + int(mm) / 60.0 + int(ss) / 3600.0
    except ValueError:
        raise ValueError(
            f"local_time_at_node must contain integer hour/minute/second "
            f"fields, got '{local_time_at_node}'"
        )
    if not (0.0 <= lsol < 24.0):
        raise ValueError(
            f"Parsed local time {lsol:.4f} h is outside [0, 24), "
            f"got '{local_time_at_node}'"
        )

    # --- validate node_type ---
    if node_type == 'ascending':
        ltan = lsol                       # LTAN  = specified time
    elif node_type == 'descending':
        ltan = (lsol + 12.0) % 24.0      # LTAN  = LTDN + 12 h
    else:
        raise ValueError(
            f"node_type must be 'ascending' or 'descending', got '{node_type}'"
        )

    # --- validate altitude ---
    if altitude < 0.0:
        raise ValueError(f"altitude must be non-negative, got {altitude} m")

    # --- Sun's right ascension at epoch (low-precision, ~0.01° accuracy) ---
    # Algorithm: Astronomical Almanac "low-precision" solar coordinates
    d = float((epoch_us - _J2000_US).astype(np.float64)) * 1e-6 / 86400.0   # days from J2000

    L_deg = (280.460 + 0.9856474 * d) % 360.0        # Sun mean longitude (deg)
    g_rad = np.radians((357.528 + 0.9856003 * d) % 360.0)  # Sun mean anomaly (rad)

    # Ecliptic longitude (rad)
    lambda_sun = np.radians(L_deg) + np.radians(
        1.915 * np.sin(g_rad) + 0.020 * np.sin(2.0 * g_rad)
    )

    # Mean obliquity of the ecliptic (rad)
    epsilon = np.radians(23.439 - 0.0000004 * d)

    # Sun's right ascension (rad), wrapped to [0, 2π)
    ra_sun = float(
        np.arctan2(np.cos(epsilon) * np.sin(lambda_sun),
                   np.cos(lambda_sun)) % (2.0 * np.pi)
    )

    # --- RAAN from LTAN and Sun's RA ---
    # Derivation: LTAN (h) = 12 + (RAAN − RA_sun) × 12/π
    raan = float((ra_sun + (ltan - 12.0) * (np.pi / 12.0)) % (2.0 * np.pi))

    # --- orbital elements ---
    a = float(central_body_radius) + float(altitude)
    i = sun_synchronous_inclination(a, 0.0, central_body_mu, central_body_j2)

    return {
        'epoch':               epoch_us,
        'a':                   a,
        'e':                   0.0,
        'i':                   i,
        'arg_p':               0.0,
        'raan':                raan,
        'ma':                  0.0,
        'central_body_mu':     float(central_body_mu),
        'central_body_j2':     float(central_body_j2),
        'central_body_radius': float(central_body_radius),
    }