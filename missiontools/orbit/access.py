import numpy as np
import numpy.typing as npt

from .frames import geodetic_to_ecef, eci_to_ecef
from .propagation import propagate_analytical


def earth_access(vecs:   npt.NDArray[np.floating],
                 lat:    float | np.floating,
                 lon:    float | np.floating,
                 alt:    float | np.floating = 0.0,
                 el_min: float | np.floating = 0.0,
                 frame:  str = 'eci',
                 t:      npt.NDArray[np.datetime64] | None = None,
                 ) -> npt.NDArray[np.bool_]:
    """Determine which positions are visible from a ground station.

    A position is considered visible when the elevation angle from the ground
    station to that position is greater than or equal to ``el_min``.

    .. note::
        Earth blockage is not explicitly checked. For a surface station with
        ``el_min >= 0``, a positive elevation angle implies the line-of-sight
        clears the Earth. Sub-zero masks or elevated ground stations may
        require an additional ray–ellipsoid intersection check.

    Parameters
    ----------
    vecs : npt.NDArray[np.floating]
        Position vectors, shape ``(N, 3)`` (m).
    lat : float | np.floating
        Ground station geodetic latitude (rad).
    lon : float | np.floating
        Ground station longitude (rad).
    alt : float | np.floating, optional
        Ground station height above the WGS84 ellipsoid (m). Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad). Defaults to 0 (above the horizon).
    frame : str, optional
        Reference frame of ``vecs``: ``'eci'`` (default) or ``'ecef'``.
    t : npt.NDArray[np.datetime64] | None, optional
        UTC/UT1 observation times as ``datetime64[us]``, shape ``(N,)``.
        Required when ``frame='eci'``; ignored when ``frame='ecef'``.

    Returns
    -------
    npt.NDArray[np.bool_]
        Boolean array of shape ``(N,)``. ``True`` where the position is
        visible from the ground station above ``el_min``.

    Raises
    ------
    ValueError
        If ``frame='eci'`` and ``t`` is ``None``, or if ``frame`` is not
        ``'eci'`` or ``'ecef'``.
    """
    if frame == 'eci':
        if t is None:
            raise ValueError("t must be provided when frame='eci'")
        vecs_ecef = eci_to_ecef(vecs, t)
    elif frame == 'ecef':
        vecs_ecef = np.atleast_2d(vecs)
    else:
        raise ValueError(f"frame must be 'eci' or 'ecef', got '{frame!r}'")

    # Ground station ECEF position
    gs_ecef = geodetic_to_ecef(lat, lon, alt)  # (3,)

    # Outward ellipsoid normal at (lat, lon) — the geodetic "up" direction.
    # Using the geodetic normal (not gs_ecef / |gs_ecef|) gives elevation
    # angles consistent with a spirit level at the ground station.
    up = np.array([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)])  # unit vector

    # Line-of-sight from ground station to each position
    los = vecs_ecef - gs_ecef                             # (N, 3)
    los_unit = los / np.linalg.norm(los, axis=1, keepdims=True)

    # Elevation = arcsin of the component of los_unit along "up"
    sin_el = los_unit @ up                                # (N,)
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))

    return el >= el_min


def earth_access_intervals(
        t_start:          np.datetime64,
        t_end:            np.datetime64,
        keplerian_params: dict,
        lat:              float | np.floating,
        lon:              float | np.floating,
        alt:              float | np.floating = 0.0,
        el_min:           float | np.floating = 0.0,
        propagator_type:  str = 'twobody',
        max_step:         np.timedelta64 = np.timedelta64(30, 's'),
        refine_tol:       np.timedelta64 = np.timedelta64(1, 's'),
        batch_size:       int = 10_000,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Find time intervals when a satellite is visible from a ground station.

    Performs a coarse scan at ``max_step`` cadence to detect access windows,
    then refines each rising/falling edge with binary search to within
    ``refine_tol``.

    .. warning::
        Passes shorter than ``max_step`` may be missed entirely. Set
        ``max_step`` to at most half the shortest expected pass duration.

    Parameters
    ----------
    t_start : np.datetime64
        Start of the search window (``datetime64[us]``).
    t_end : np.datetime64
        End of the search window (``datetime64[us]``).
    keplerian_params : dict
        Orbital elements at epoch. Must contain the keys ``epoch``, ``a``,
        ``e``, ``i``, ``arg_p``, ``raan``, ``ma``. Optionally
        ``central_body_mu``, ``central_body_j2``, ``central_body_radius``.
    lat : float | np.floating
        Ground station geodetic latitude (rad).
    lon : float | np.floating
        Ground station longitude (rad).
    alt : float | np.floating, optional
        Ground station height above the WGS84 ellipsoid (m). Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad). Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Maximum coarse scan step size. Defaults to 30 s.
    refine_tol : np.timedelta64, optional
        Binary-search convergence tolerance for edge times. Defaults to 1 s.
    batch_size : int, optional
        Number of scan steps per propagation batch. Limits peak memory usage
        to roughly ``batch_size × 24`` bytes. Defaults to 10 000.

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        List of ``(start, end)`` pairs in ``datetime64[us]``, one per
        continuous access window. Empty list if no access occurs.
    """
    t_start = np.asarray(t_start, dtype='datetime64[us]')
    t_end   = np.asarray(t_end,   dtype='datetime64[us]')

    total_us = int((t_end   - t_start) / np.timedelta64(1, 'us'))
    step_us  = int(max_step   / np.timedelta64(1, 'us'))
    tol_us   = int(refine_tol / np.timedelta64(1, 'us'))

    if total_us <= 0 or step_us <= 0:
        return []

    # Scan offsets (µs from t_start), always including t_end exactly
    offsets = np.arange(0, total_us + 1, step_us, dtype=np.int64)
    if offsets[-1] < total_us:
        offsets = np.append(offsets, np.int64(total_us))
    n_total = len(offsets)

    # --- helpers (closures over outer params) ---

    def t_at(off: int) -> np.datetime64:
        return t_start + np.timedelta64(int(off), 'us')

    def access_at(off: int) -> bool:
        t_arr = np.array([t_at(off)])
        r, _ = propagate_analytical(t_arr, **keplerian_params,
                                    type=propagator_type)
        return bool(earth_access(r, lat, lon, alt, el_min,
                                 frame='eci', t=t_arr)[0])

    def refine(lo: int, hi: int, rising: bool) -> int:
        """Binary search: converge on the transition to within tol_us."""
        tol = max(tol_us, 1)          # guard against zero-tolerance infinite loop
        while hi - lo > tol:
            mid = lo + (hi - lo) // 2
            if rising == access_at(mid):
                hi = mid
            else:
                lo = mid
        return hi if rising else lo

    # --- batched coarse scan ---

    intervals: list[tuple[np.datetime64, np.datetime64]] = []
    interval_start_us: int | None = None
    prev_flag: bool | None = None
    prev_offset: int = 0

    for batch_start in range(0, n_total, batch_size):
        batch_end  = min(batch_start + batch_size, n_total)
        batch_offs = offsets[batch_start:batch_end]

        t_batch = t_start + batch_offs.astype('timedelta64[us]')
        r, _    = propagate_analytical(t_batch, **keplerian_params,
                                       type=propagator_type)
        flags   = earth_access(r, lat, lon, alt, el_min,
                                frame='eci', t=t_batch)

        if prev_flag is None:
            # Bootstrap from the very first scan point
            prev_flag   = bool(flags[0])
            prev_offset = int(batch_offs[0])
            if prev_flag:
                interval_start_us = prev_offset
            batch_offs = batch_offs[1:]
            flags      = flags[1:]

        if len(batch_offs) == 0:
            continue

        # Vectorised transition detection within this chunk
        prev_and_flags = np.concatenate([[prev_flag], flags])   # (B+1,)
        change_k = np.where(prev_and_flags[:-1] != prev_and_flags[1:])[0]

        for k in change_k:
            lo     = prev_offset if k == 0 else int(batch_offs[k - 1])
            hi     = int(batch_offs[k])
            rising = not bool(prev_and_flags[k])   # prev state False → rising

            if rising:
                interval_start_us = refine(lo, hi, rising=True)
            else:
                t_fall = refine(lo, hi, rising=False)
                if interval_start_us is not None:
                    intervals.append((t_at(interval_start_us), t_at(t_fall)))
                interval_start_us = None

        prev_flag   = bool(flags[-1])
        prev_offset = int(batch_offs[-1])

    # Close any interval still open at t_end
    if interval_start_us is not None:
        intervals.append((t_at(interval_start_us), t_end))

    return intervals
