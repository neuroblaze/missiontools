#!/usr/bin/env python3
"""Conditional coverage Cesium 3D visualization.

Visualises the nadir-imaging-with-ground-station-tracking scenario from
``conditional_coverage.ipynb`` using the Cesium 3D globe viewer.

The spacecraft alternates between nadir imaging (default) and Svalbard
tracking (when the ground station is in view).  Attitude mode switching
is visible on the 3D model; the sensor FOV is also rendered.

Requires ``czml3``, ``pywebview``, and ``cartopy``::

    pip install missiontools[cesium,plot]
"""

import numpy as np

from missiontools import (
    Spacecraft,
    ConicSensor,
    AoI,
    GroundStation,
    FixedAttitudeLaw,
    TrackAttitudeLaw,
    ConditionAttitudeLaw,
    SpaceGroundAccessCondition,
)
from missiontools.cesium import CesiumViewer


T0 = np.datetime64("2025-06-01T00:00:00", "us")
T1 = T0 + np.timedelta64(24 * 3600, "s")
STEP = np.timedelta64(30, "s")

ALT_KM = 550.0
LTDN = "10:30"

GS_SVALBARD = GroundStation(lat=78.2, lon=15.6)

AOI_EU = AoI.from_region(
    lat_min_deg=27.0,
    lat_max_deg=72.0,
    lon_min_deg=-32.0,
    lon_max_deg=45.0,
)

SC_COLOR = (0, 200, 255, 255)


def main() -> None:
    sc = Spacecraft.sunsync(
        altitude_km=ALT_KM,
        node_solar_time=LTDN,
        epoch=T0,
    )

    gs_visible = SpaceGroundAccessCondition(sc, GS_SVALBARD, el_min_deg=5.0)
    attitude_law = ConditionAttitudeLaw(
        default_attitude=FixedAttitudeLaw.nadir(),
        condition_attitudes=[(gs_visible, TrackAttitudeLaw(GS_SVALBARD))],
    )
    sc.attitude_law = attitude_law

    sensor = ConicSensor(15.0, body_vector=[0, 0, 1], condition=~gs_visible)
    sc.add_sensor(sensor)

    viewer = CesiumViewer(title="Conditional Coverage — Nadir + Svalbard Tracking")
    viewer.add_spacecraft(
        sc,
        T0,
        T1,
        STEP,
        color=SC_COLOR,
        label="SSO-550",
        show_sensors=True,
    )
    viewer.add_ground_station(GS_SVALBARD, label="Svalbard")
    viewer.add_aoi(AOI_EU, color=(255, 80, 80, 80), label="EU Region")
    viewer.show()


if __name__ == "__main__":
    main()
