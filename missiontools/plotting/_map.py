"""
missiontools.plotting._map
==========================
Shared Cartopy map-setup helpers.
"""
from __future__ import annotations

import numpy as np


def _try_cartopy():
    """Import cartopy or raise a clear ImportError."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        raise ImportError(
            "missiontools.plotting requires cartopy. "
            "Install it with:  pip install missiontools[plot]"
        ) from None


def _new_map_ax(ax=None, projection=None):
    """Return a GeoAxes decorated with coastlines, borders, and gridlines.

    If *ax* is ``None``, a new figure and GeoAxes are created using
    *projection* (default: ``ccrs.PlateCarree()`` — WGS-84 equirectangular).
    The axes extent is set to the full Earth.

    Parameters
    ----------
    ax : GeoAxes, optional
        Existing axes to decorate.  If provided, *projection* is ignored.
    projection : cartopy CRS, optional
        Map projection for the new axes.  Default ``ccrs.PlateCarree()``.

    Returns
    -------
    GeoAxes
    """
    ccrs, cfeature = _try_cartopy()
    import matplotlib.pyplot as plt

    if projection is None:
        projection = ccrs.PlateCarree()

    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': projection})

    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    return ax


def _set_extent(ax, lat, lon, factor: float = 1.5) -> None:
    """Auto-window the axes to *factor* × the data lat/lon bounding box.

    Pads each axis by ``(factor - 1) / 2 × range``, then clamps to valid
    Earth bounds and calls ``ax.set_extent``.

    Parameters
    ----------
    ax : GeoAxes
    lat : ndarray, degrees
    lon : ndarray, degrees
    factor : float
        Window size as a multiple of the data range (default 1.5).
    """
    ccrs, _ = _try_cartopy()

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())

    # Ensure at least 1° of range in each axis before padding
    pad_lat = (factor - 1) / 2 * max(lat_max - lat_min, 1.0)
    pad_lon = (factor - 1) / 2 * max(lon_max - lon_min, 1.0)

    extent = [
        max(lon_min - pad_lon, -180.0),
        min(lon_max + pad_lon,  180.0),
        max(lat_min - pad_lat,  -90.0),
        min(lat_max + pad_lat,   90.0),
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
