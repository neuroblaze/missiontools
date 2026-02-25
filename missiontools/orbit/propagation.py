import numpy as np
import numpy.typing as npt
from datetime import datetime
from .constants import EARTH_MU, EARTH_J2, EARTH_MEAN_RADIUS


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

        # Vallado eq. 9-37
        raan_dot = (-3.0*mean_motion*central_body_radius**2*central_body_j2) / \
                   (2.0*p**2) * np.cos(i)
        raan = raan + raan_dot*t_e

        # Vallado eq. 9-39
        arg_p_dot = (3.0*mean_motion*central_body_radius**2*central_body_j2) / \
                    (4.0*p**2) * (4.0-5.0*np.sin(i)**2)
        arg_p = arg_p + arg_p_dot*t_e

        # Vallado eq. 9-41
        ma_dot = (-3.0*mean_motion*central_body_radius**2*central_body_j2*np.sqrt(1-e**2)) / \
                 (4.0*p**2) * (3.0*np.sin(i) - 2)
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