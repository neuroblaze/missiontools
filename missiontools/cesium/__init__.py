"""
missiontools.cesium
===================
CesiumJS-based 3D globe visualization for missiontools objects.

This subpackage requires ``czml3`` and ``pywebview``.  Install with::

    pip install missiontools[cesium]

Quick start
-----------
::

    import numpy as np
    from missiontools import Spacecraft, GroundStation
    from missiontools.cesium import CesiumViewer

    sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
    gs = GroundStation(lat=78.2, lon=15.6)

    t0 = np.datetime64('2025-01-01', 'us')
    t1 = np.datetime64('2025-01-02', 'us')

    viewer = CesiumViewer()
    viewer.add_spacecraft(sc, t0, t1)
    viewer.add_ground_station(gs, label='Svalbard')
    viewer.show()

Classes
-------
:class:`CesiumViewer`
    Collects domain objects and renders them in an interactive pywebview
    window backed by CesiumJS.
"""

from ._viewer import CesiumViewer

__all__ = ["CesiumViewer"]
