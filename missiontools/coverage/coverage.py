import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as _MplPath

from ..orbit.frames import geodetic_to_ecef, eci_to_ecef, lvlh_to_eci
from ..orbit.propagation import propagate_analytical
from ..orbit.constants import EARTH_MEAN_RADIUS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Equal-area Fibonacci lattice: n points on the unit sphere (radians)."""
    if n == 1:
        return np.array([0.0]), np.array([0.0])
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i   = np.arange(n, dtype=np.float64)
    lat = np.arcsin(np.clip(2.0 * i / (n - 1) - 1.0, -1.0, 1.0))
    lon = (2.0 * np.pi * i / phi) % (2.0 * np.pi) - np.pi   # → (−π, π]
    return lat, lon


def _pip(polygon: npt.NDArray,
         lat:     npt.NDArray,
         lon:     npt.NDArray) -> npt.NDArray[np.bool_]:
    """Planar point-in-polygon test in lat/lon space (radians)."""
    # MplPath uses (x, y) = (lon, lat)
    path = _MplPath(polygon[:, ::-1])
    return path.contains_points(np.column_stack([lon, lat]))


def _build_gs(lat: npt.NDArray,
              lon: npt.NDArray,
              alt: float | npt.NDArray,
              ) -> tuple[npt.NDArray, npt.NDArray]:
    """Return ground-point ECEF positions (M,3) and geodetic up-vectors (M,3)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    gs_ecef = geodetic_to_ecef(lat, lon, alt)           # (M, 3)
    up = np.stack([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)], axis=-1)                # (M, 3)
    return gs_ecef, up


def _pointing_ecef(pointing_lvlh: npt.NDArray,   # (3,) unit vector in LVLH
                   r_eci:         npt.NDArray,   # (T, 3)
                   v_eci:         npt.NDArray,   # (T, 3)
                   t_arr:         npt.NDArray,   # (T,) datetime64[us]
                   ) -> npt.NDArray:             # (T, 3)
    """Convert a fixed LVLH pointing direction to ECEF at each timestep."""
    T     = len(t_arr)
    p_eci = lvlh_to_eci(np.tile(pointing_lvlh, (T, 1)), r_eci, v_eci)
    return eci_to_ecef(p_eci, t_arr)


def _visibility(r_eci:          npt.NDArray,         # (T, 3)
                t_arr:          npt.NDArray,         # (T,) datetime64[us]
                gs_ecef:        npt.NDArray,         # (M, 3)
                up:             npt.NDArray,         # (M, 3)
                sin_el_min:     float,
                pointing_ecef:  npt.NDArray | None = None,  # (T, 3)
                cos_fov:        float | None = None,
                ) -> npt.NDArray[np.bool_]:          # (T, M)
    """Vectorised visibility: T satellite positions × M ground points.

    When ``pointing_ecef`` and ``cos_fov`` are supplied (pre-converted once per
    batch by the caller) a FOV cone constraint is ANDed with the elevation mask.
    """
    r_ecef  = eci_to_ecef(r_eci, t_arr)                               # (T, 3)
    los     = r_ecef[:, np.newaxis, :] - gs_ecef[np.newaxis, :, :]   # (T, M, 3)
    norm    = np.maximum(np.linalg.norm(los, axis=2, keepdims=True), 1e-10)
    los_hat = los / norm                                               # (T, M, 3)
    sin_el  = np.einsum('tmi,mi->tm', los_hat, up)                    # (T, M)
    vis     = sin_el >= sin_el_min                                     # (T, M)

    if pointing_ecef is not None:
        # dot(sat→ground, pointing) = dot(-los_hat, pointing_ecef)
        fov_cos = np.einsum('tmi,ti->tm', -los_hat, pointing_ecef)    # (T, M)
        vis    &= fov_cos >= cos_fov

    return vis


def _make_offsets(t_start:  np.datetime64,
                  t_end:    np.datetime64,
                  max_step: np.timedelta64,
                  ) -> tuple[npt.NDArray[np.int64], np.datetime64]:
    """Integer µs offsets from t_start, always including t_end."""
    t_start   = np.asarray(t_start, dtype='datetime64[us]')
    t_end     = np.asarray(t_end,   dtype='datetime64[us]')
    total_us  = int((t_end   - t_start) / np.timedelta64(1, 'us'))
    step_us   = int(max_step / np.timedelta64(1, 'us'))
    if total_us <= 0 or step_us <= 0:
        return np.array([], dtype=np.int64), t_start
    offs = np.arange(0, total_us + 1, step_us, dtype=np.int64)
    if offs[-1] != total_us:
        offs = np.append(offs, np.int64(total_us))
    return offs, t_start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_aoi(
        polygon: npt.NDArray[np.floating],
        n:       int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample *n* approximately equal-area points inside an AoI polygon.

    Uses a Fibonacci lattice to generate a near-uniform global distribution,
    then filters to the points inside the polygon.  The returned count may
    differ slightly from *n* depending on AoI shape; use a larger *n* for
    denser or irregular regions.

    .. note::
        The polygon test is planar (lat/lon space in radians).  Results are
        accurate for regions that do not span the anti-meridian or enclose a
        pole.  Split such regions into separate polygons and combine samples.

    Parameters
    ----------
    polygon : npt.NDArray[np.floating]
        Polygon vertices, shape ``(V, 2)``, each row ``[lat, lon]`` in
        **radians**.
    n : int
        Target number of sample points.

    Returns
    -------
    lat : npt.NDArray[np.floating]
        Sample latitudes (rad), shape ``(M,)``.
    lon : npt.NDArray[np.floating]
        Sample longitudes (rad), shape ``(M,)``.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (V, 2) with columns [lat, lon]")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    # Estimate AoI fraction from a pilot lattice
    n_pilot = max(n * 10, 5_000)
    lat_p, lon_p = _fibonacci_sphere(n_pilot)
    frac = float(_pip(polygon, lat_p, lon_p).mean())

    if frac < 1e-6:
        raise ValueError(
            "AoI polygon encloses too few global lattice points — "
            "check that coordinates are in radians and the polygon is not "
            "degenerate."
        )

    # Generate enough global points to get approximately n inside
    n_global = int(np.ceil(n / frac * 1.3))
    lat_all, lon_all = _fibonacci_sphere(n_global)
    inside  = _pip(polygon, lat_all, lon_all)
    lat_in  = lat_all[inside]
    lon_in  = lon_all[inside]

    # Evenly subsample if we ended up with more than n points
    if len(lat_in) > n:
        idx    = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def sample_region(
        lat_min:       float | None = None,
        lat_max:       float | None = None,
        lon_min:       float | None = None,
        lon_max:       float | None = None,
        point_density: float = 1e11,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample an approximately equal-area Fibonacci lattice over a lat/lon band.

    A convenience wrapper around the Fibonacci lattice for rectangular
    regions.  The number of points is derived from the spherical zone area
    and the requested ``point_density``, so the grid automatically becomes
    denser for small regions and sparser for large ones.

    Unlike :func:`sample_aoi`, this function filters the lattice directly by
    coordinate bounds, so it handles anti-meridian-crossing regions and global
    or banded coverage correctly.

    Parameters
    ----------
    lat_min : float | None, optional
        Southern boundary (rad).  ``None`` extends to the South Pole (−π/2).
    lat_max : float | None, optional
        Northern boundary (rad).  ``None`` extends to the North Pole (+π/2).
    lon_min : float | None, optional
        Western boundary (rad).  Must be paired with ``lon_max``; ``None``
        (together with ``lon_max=None``) includes all longitudes.
    lon_max : float | None, optional
        Eastern boundary (rad).  Must be paired with ``lon_min``; ``None``
        (together with ``lon_min=None``) includes all longitudes.
        May be less than ``lon_min`` to indicate a region that crosses the
        anti-meridian (e.g. ``lon_min=np.radians(170)``,
        ``lon_max=np.radians(-170)``).
    point_density : float, optional
        Approximate area represented by each sample point (m²).
        Defaults to 1×10¹¹ m² (~100 000 km² per point, ~5 100 points
        globally).

    Returns
    -------
    lat : npt.NDArray[np.floating]
        Sample latitudes (rad), shape ``(M,)``.
    lon : npt.NDArray[np.floating]
        Sample longitudes (rad), shape ``(M,)``.

    Raises
    ------
    ValueError
        If exactly one of ``lon_min`` / ``lon_max`` is ``None``, if
        ``lat_min >= lat_max``, or if ``point_density`` is not positive.

    Examples
    --------
    Global coverage at ~100 000 km²/point::

        lat, lon = sample_region()

    Europe (approximate)::

        lat, lon = sample_region(np.radians(35), np.radians(70),
                                 np.radians(-10), np.radians(40),
                                 point_density=1e9)

    Pacific Ocean band crossing the anti-meridian::

        lat, lon = sample_region(np.radians(-30), np.radians(30),
                                 np.radians(150), np.radians(-120),
                                 point_density=1e10)
    """
    # --- validate longitude pairing ---
    if (lon_min is None) != (lon_max is None):
        raise ValueError(
            "lon_min and lon_max must both be None (all longitudes) or both "
            "be specified; got lon_min={} lon_max={}".format(lon_min, lon_max)
        )

    if point_density <= 0:
        raise ValueError(
            f"point_density must be positive, got {point_density}"
        )

    # --- resolve defaults ---
    lat_lo   = float(lat_min) if lat_min is not None else -np.pi / 2.0
    lat_hi   = float(lat_max) if lat_max is not None else  np.pi / 2.0
    full_lon = lon_min is None

    if lat_lo >= lat_hi:
        raise ValueError(
            f"lat_min ({lat_lo:.6f} rad) must be less than "
            f"lat_max ({lat_hi:.6f} rad)"
        )

    lon_lo = float(lon_min) if lon_min is not None else 0.0
    lon_hi = float(lon_max) if lon_max is not None else 0.0
    antimeridian = (not full_lon) and (lon_lo > lon_hi)

    # --- compute AoI area (spherical zone formula) ---
    # Area = 2π R² Δ(sin lat) × (lon span / 2π)
    if full_lon:
        lon_frac = 1.0
    elif antimeridian:
        lon_frac = (2.0 * np.pi - (lon_lo - lon_hi)) / (2.0 * np.pi)
    else:
        lon_frac = (lon_hi - lon_lo) / (2.0 * np.pi)

    area = (4.0 * np.pi * EARTH_MEAN_RADIUS**2
            * (np.sin(lat_hi) - np.sin(lat_lo)) / 2.0
            * lon_frac)

    n = max(1, int(np.round(area / point_density)))

    # --- generate a global Fibonacci lattice and filter ---
    # Oversample globally so that after filtering we have ≥ n points.
    area_fraction = area / (4.0 * np.pi * EARTH_MEAN_RADIUS**2)
    n_global = max(n * 5, int(np.ceil(n / area_fraction * 1.3)))
    lat_all, lon_all = _fibonacci_sphere(n_global)

    # Latitude filter
    lat_ok = (lat_all >= lat_lo) & (lat_all <= lat_hi)

    # Longitude filter
    if full_lon:
        lon_ok = np.ones(n_global, dtype=np.bool_)
    elif antimeridian:
        lon_ok = (lon_all >= lon_lo) | (lon_all <= lon_hi)
    else:
        lon_ok = (lon_all >= lon_lo) & (lon_all <= lon_hi)

    lat_in = lat_all[lat_ok & lon_ok]
    lon_in = lon_all[lat_ok & lon_ok]

    # Evenly subsample if we have more points than requested
    if len(lat_in) > n:
        idx    = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def coverage_fraction(
        lat:               npt.NDArray[np.floating],
        lon:               npt.NDArray[np.floating],
        keplerian_params:  dict,
        t_start:           np.datetime64,
        t_end:             np.datetime64,
        alt:               float | np.floating = 0.0,
        el_min:            float | np.floating = 0.0,
        propagator_type:   str = 'twobody',
        max_step:          np.timedelta64 = np.timedelta64(30, 's'),
        batch_size:        int = 1_000,
        fov_pointing_lvlh: npt.NDArray | None = None,
        fov_half_angle:    float | None = None,
) -> dict:
    """Compute instantaneous and cumulative coverage fraction over a time window.

    For each sample epoch, the *instantaneous* fraction is the proportion of
    ground points with the satellite above ``el_min``.  The *cumulative*
    fraction is the proportion of ground points seen **at least once** up to
    that epoch.

    Parameters
    ----------
    lat, lon : npt.NDArray[np.floating]
        Ground-point latitudes/longitudes (rad), shape ``(M,)``.  Typically
        from :func:`sample_aoi`.
    keplerian_params : dict
        Orbital elements dict, e.g. from :func:`~missiontools.orbit.propagation.sun_synchronous_orbit`.
    t_start, t_end : np.datetime64
        Analysis window (``datetime64[us]``).
    alt : float | np.floating, optional
        Ground-point altitude above WGS84 (m).  Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad).  Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Scan time step.  Defaults to 30 s.
    batch_size : int, optional
        Time steps per propagation batch.  Defaults to 1 000.

    Returns
    -------
    dict
        ``t`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

        ``fraction`` : ``(N,)`` float — instantaneous coverage fraction.

        ``cumulative`` : ``(N,)`` float — cumulative coverage fraction.

        ``mean_fraction`` : float — time-averaged instantaneous fraction.

        ``final_cumulative`` : float — fraction of points covered ≥ once.
    """
    # --- FOV validation & setup --------------------------------------------------
    _fov_given = (fov_pointing_lvlh is not None, fov_half_angle is not None)
    if any(_fov_given) and not all(_fov_given):
        raise ValueError("fov_pointing_lvlh and fov_half_angle must both be "
                         "provided together")
    use_fov = all(_fov_given)
    if use_fov:
        pointing_lvlh_norm = (np.asarray(fov_pointing_lvlh, dtype=np.float64)
                              / np.linalg.norm(fov_pointing_lvlh))
        cos_fov = float(np.cos(fov_half_angle))
    # --------------------------------------------------------------------------

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M   = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)
    sin_el_min  = float(np.sin(el_min))

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    if N == 0:
        empty = np.array([], dtype=np.float32)
        return {
            't': np.array([], dtype='datetime64[us]'),
            'fraction': empty, 'cumulative': empty,
            'mean_fraction': float('nan'), 'final_cumulative': float('nan'),
        }

    t_out    = t_start_us + offs.astype('timedelta64[us]')   # (N,)
    frac_out = np.empty(N, dtype=np.float32)
    cum_out  = np.empty(N, dtype=np.float32)

    ever_covered = np.zeros(M, dtype=np.bool_)
    n_covered    = 0

    for b0 in range(0, N, batch_size):
        b1      = min(b0 + batch_size, N)
        t_batch = t_out[b0:b1]
        r, v    = propagate_analytical(t_batch, **keplerian_params,
                                       type=propagator_type)
        pt_ecef = (_pointing_ecef(pointing_lvlh_norm, r, v, t_batch)
                   if use_fov else None)
        vis     = _visibility(r, t_batch, gs_ecef, up, sin_el_min,
                              pointing_ecef=pt_ecef,
                              cos_fov=cos_fov if use_fov else None)  # (T, M)

        frac_out[b0:b1] = vis.mean(axis=1)

        for local_t in range(b1 - b0):
            new = vis[local_t] & ~ever_covered
            if new.any():
                ever_covered |= new
                n_covered    += int(new.sum())
            cum_out[b0 + local_t] = n_covered / M

    return {
        't':                t_out,
        'fraction':         frac_out,
        'cumulative':       cum_out,
        'mean_fraction':    float(np.mean(frac_out)),
        'final_cumulative': float(cum_out[-1]),
    }


def revisit_time(
        lat:               npt.NDArray[np.floating],
        lon:               npt.NDArray[np.floating],
        keplerian_params:  dict,
        t_start:           np.datetime64,
        t_end:             np.datetime64,
        alt:               float | np.floating = 0.0,
        el_min:            float | np.floating = 0.0,
        propagator_type:   str = 'twobody',
        max_step:          np.timedelta64 = np.timedelta64(30, 's'),
        batch_size:        int = 1_000,
        fov_pointing_lvlh: npt.NDArray | None = None,
        fov_half_angle:    float | None = None,
) -> dict:
    """Compute per-point revisit time statistics over a time window.

    The *revisit time* for a ground point is the gap between loss of signal
    (LOS, last visible step) and the next acquisition of signal (AOS, first
    visible step on the following pass).  The initial gap from ``t_start`` to
    the first AOS is not included.

    .. note::
        Accuracy is limited to ``max_step``.  For exact edge times on
        individual points of interest use
        :func:`~missiontools.orbit.access.earth_access_intervals`.

    Parameters
    ----------
    lat, lon : npt.NDArray[np.floating]
        Ground-point latitudes/longitudes (rad), shape ``(M,)``.
    keplerian_params : dict
        Orbital elements dict.
    t_start, t_end : np.datetime64
        Analysis window.
    alt : float | np.floating, optional
        Ground-point altitude above WGS84 (m).  Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad).  Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` or ``'j2'``.
    max_step : np.timedelta64, optional
        Scan step.  Defaults to 30 s.
    batch_size : int, optional
        Time steps per propagation batch.  Defaults to 1 000.

    Returns
    -------
    dict
        ``max_revisit`` : ``(M,)`` float — per-point max revisit time (s).
        ``nan`` for points accessed fewer than twice.

        ``mean_revisit`` : ``(M,)`` float — per-point mean revisit time (s).
        ``nan`` for points accessed fewer than twice.

        ``global_max`` : float — worst-case revisit time across all points (s).

        ``global_mean`` : float — mean of per-point mean revisit times (s).
    """
    # --- FOV validation & setup --------------------------------------------------
    _fov_given = (fov_pointing_lvlh is not None, fov_half_angle is not None)
    if any(_fov_given) and not all(_fov_given):
        raise ValueError("fov_pointing_lvlh and fov_half_angle must both be "
                         "provided together")
    use_fov = all(_fov_given)
    if use_fov:
        pointing_lvlh_norm = (np.asarray(fov_pointing_lvlh, dtype=np.float64)
                              / np.linalg.norm(fov_pointing_lvlh))
        cos_fov = float(np.cos(fov_half_angle))
    # --------------------------------------------------------------------------

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M   = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)
    sin_el_min  = float(np.sin(el_min))

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    _nan = np.full(M, np.nan)
    if N == 0:
        return {'max_revisit': _nan, 'mean_revisit': _nan,
                'global_max': float('nan'), 'global_mean': float('nan')}

    t_out = t_start_us + offs.astype('timedelta64[us]')  # (N,)

    # Per-point state (all µs offsets from t_start)
    in_access    = np.zeros(M, dtype=np.bool_)   # current access state
    had_los      = np.zeros(M, dtype=np.bool_)   # seen at least one LOS?
    prev_los_us  = np.zeros(M, dtype=np.int64)   # offset of most recent LOS
    max_gap_us   = np.zeros(M, dtype=np.int64)
    total_gap_us = np.zeros(M, dtype=np.int64)
    gap_count    = np.zeros(M, dtype=np.int64)

    for b0 in range(0, N, batch_size):
        b1      = min(b0 + batch_size, N)
        b_offs  = offs[b0:b1]                          # (T,) µs offsets
        t_batch = t_out[b0:b1]
        r, v    = propagate_analytical(t_batch, **keplerian_params,
                                       type=propagator_type)
        pt_ecef = (_pointing_ecef(pointing_lvlh_norm, r, v, t_batch)
                   if use_fov else None)
        vis     = _visibility(r, t_batch, gs_ecef, up, sin_el_min,
                              pointing_ecef=pt_ecef,
                              cos_fov=cos_fov if use_fov else None)  # (T, M)

        # Detect transitions: prepend last-known state as row 0
        augmented = np.vstack([in_access[np.newaxis, :],
                               vis.astype(np.int8)]).astype(np.int8)  # (T+1, M)
        diffs = np.diff(augmented, axis=0)                             # (T,  M)
        rising_t,  rising_m  = np.where(diffs > 0)
        falling_t, falling_m = np.where(diffs < 0)

        # Merge into a time-ordered list of (offset_us, m, is_rising)
        all_offs = np.concatenate([b_offs[rising_t],  b_offs[falling_t]])
        all_m    = np.concatenate([rising_m,           falling_m])
        all_rise = np.concatenate([np.ones(len(rising_t),  dtype=np.bool_),
                                   np.zeros(len(falling_t), dtype=np.bool_)])
        order    = np.argsort(all_offs, kind='stable')

        for k in order:
            off_us   = int(all_offs[k])
            m        = int(all_m[k])
            is_rising = bool(all_rise[k])

            if is_rising:
                if had_los[m]:
                    gap = off_us - prev_los_us[m]
                    if gap > max_gap_us[m]:
                        max_gap_us[m] = gap
                    total_gap_us[m] += gap
                    gap_count[m]    += 1
            else:
                prev_los_us[m] = off_us
                had_los[m]     = True

        in_access[:] = vis[-1]

    # Convert µs → seconds; NaN for points with fewer than 2 accesses.
    # Guard gap_count==0 with np.where before dividing to avoid a spurious
    # RuntimeWarning (the guarded branch is discarded by the outer np.where).
    has_gaps = gap_count > 0
    safe_count = np.where(has_gaps, gap_count, 1)
    max_rev  = np.where(has_gaps, max_gap_us              / 1e6, np.nan)
    mean_rev = np.where(has_gaps, total_gap_us / safe_count / 1e6, np.nan)

    return {
        'max_revisit':  max_rev,
        'mean_revisit': mean_rev,
        'global_max':   float(np.nanmax(max_rev))  if has_gaps.any() else float('nan'),
        'global_mean':  float(np.nanmean(mean_rev)) if has_gaps.any() else float('nan'),
    }
