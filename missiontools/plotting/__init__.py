"""
missiontools.plotting
=====================
Cartopy-backed visualisation helpers.

Functions
---------
plot_ground_track
    Spacecraft groundtrack on an Earth map.
plot_coverage_map
    Interpolated heatmap of per-point coverage values on an Earth map.

Notes
-----
This subpackage requires ``cartopy``.  Install it with::

    pip install missiontools[plot]
"""
from .ground_track import plot_ground_track
from .coverage_map import plot_coverage_map

__all__ = ['plot_ground_track', 'plot_coverage_map']
