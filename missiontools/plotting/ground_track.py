"""
missiontools.plotting.ground_track
===================================
Spacecraft groundtrack visualisation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._map import _try_cartopy, _new_map_ax, _set_extent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ecef_to_latlon(
        r_ecef: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert ECEF positions to geodetic latitude and longitude.

    Uses a spherical Earth model, which is sufficient for ground-track
    visualisation.

    Parameters
    ----------
    r_ecef : ndarray, shape (N, 3)
        ECEF position vectors (any length unit).

    Returns
    -------
    lat : ndarray, shape (N,)
        Geodetic latitude (degrees), range ``[-90, 90]``.
    lon : ndarray, shape (N,)
        Longitude (degrees), range ``(-180, 180]``.
    """
    x, y, z = r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return lat, lon


def _split_antimeridian(
        lat: npt.NDArray[np.floating],
        lon: npt.NDArray[np.floating],
) -> list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    """Split a lat/lon track at antimeridian crossings.

    Each crossing (|Δlon| > 180°) causes a wraparound artefact when
    plotted as a continuous line.  This function splits the track into
    segments that can each be plotted safely.

    Parameters
    ----------
    lat, lon : ndarray, shape (N,)
        Track latitude and longitude (degrees).

    Returns
    -------
    list of (lat_seg, lon_seg) tuples
        One tuple per continuous segment.
    """
    splits  = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    indices = np.concatenate([[0], splits, [len(lon)]])
    return [(lat[a:b], lon[a:b]) for a, b in zip(indices, indices[1:])]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_ground_track(
        spacecraft,
        t_start: np.datetime64,
        t_end:   np.datetime64,
        step:    np.timedelta64 = np.timedelta64(30, 's'),
        *,
        ax=None,
        projection=None,
        auto_window:      bool       = False,
        color:            str        = 'tab:blue',
        linewidth:        float      = 1.0,
        label:            str | None = None,
        add_start_marker: bool       = True,
):
    """Plot the spacecraft groundtrack on an Earth map.

    Parameters
    ----------
    spacecraft : Spacecraft
        Spacecraft whose orbit to propagate.
    t_start : np.datetime64
        Start of the analysis window.
    t_end : np.datetime64
        End of the analysis window (inclusive).
    step : np.timedelta64
        Propagation step (default 30 s).
    ax : GeoAxes, optional
        Existing Cartopy GeoAxes to draw on.  A new figure is created if
        ``None``.
    projection : cartopy CRS, optional
        Map projection for the new axes.  Default
        ``ccrs.PlateCarree()`` (WGS-84 equirectangular).  Ignored if *ax*
        is provided.
    auto_window : bool
        If ``True``, set the axes extent to 1.5× the lat/lon range of
        the groundtrack.
    color : str
        Track colour (matplotlib colour spec).
    linewidth : float
        Track line width.
    label : str, optional
        Legend label for the track line.
    add_start_marker : bool
        If ``True``, draw a filled circle at the initial sub-satellite
        point.

    Returns
    -------
    GeoAxes
        The axes on which the track was drawn.

    Examples
    --------
    ::

        import numpy as np
        from missiontools import Spacecraft
        from missiontools.plotting import plot_ground_track

        sc = Spacecraft(a=6_771_000., e=0., i=np.radians(51.6),
                        raan=0., arg_p=0., ma=0.,
                        epoch=np.datetime64('2025-01-01', 'us'))

        t0 = np.datetime64('2025-01-01', 'us')
        ax = plot_ground_track(sc, t0, t0 + np.timedelta64(5400, 's'))
    """
    from ..orbit.frames import eci_to_ecef

    ccrs, _ = _try_cartopy()
    ax = _new_map_ax(ax, projection)

    state  = spacecraft.propagate(t_start, t_end, step)
    r_ecef = eci_to_ecef(state['r'], state['t'])
    lat, lon = _ecef_to_latlon(r_ecef)

    if auto_window:
        _set_extent(ax, lat, lon)

    segs = _split_antimeridian(lat, lon)
    for i, (seg_lat, seg_lon) in enumerate(segs):
        ax.plot(
            seg_lon, seg_lat,
            transform=ccrs.PlateCarree(),
            color=color,
            linewidth=linewidth,
            label=label if i == 0 else None,
        )

    if add_start_marker and len(lat):
        ax.plot(
            lon[0], lat[0], 'o',
            color=color,
            markersize=5,
            transform=ccrs.PlateCarree(),
        )

    return ax
