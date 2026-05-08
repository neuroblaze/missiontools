#!/usr/bin/env python3
"""3-satellite SSO constellation visualization over the United States.

Demonstrates the ``missiontools.cesium`` 3D globe viewer with:
- Three sun-synchronous spacecraft in a plane (phased 120° apart)
- Each carries a nadir-pointing 10° × 40° rectangular sensor
- AoI: United States (CONUS)
- One ground station: White Sands, NM

Requires ``czml3`` and ``pywebview``::

    pip install missiontools[cesium]
"""

import numpy as np

from missiontools import (
    Spacecraft,
    GroundStation,
    AoI,
    RectangularSensor,
    FixedAttitudeLaw,
)
from missiontools.cesium import CesiumViewer


T0 = np.datetime64("2025-06-01T00:00:00", "us")
T1 = np.datetime64("2025-06-01T06:00:00", "us")
STEP = np.timedelta64(30, "s")

ALT_KM = 550.0
LTDN = "10:30"

GS_WHITE_SANDS = GroundStation(lat=32.38, lon=-106.48, alt=1300.0)

AOI_US = (
    AoI.from_geography("US") - AoI.from_geography("US-AK") - AoI.from_geography("US-HI")
)

COLORS = [
    (0, 180, 255, 255),
    (255, 100, 50, 255),
    (80, 255, 120, 255),
]


def build_constellation(n: int = 3) -> list[Spacecraft]:
    """Create *n* SSO spacecraft phased equally around one orbital plane."""
    sats = []
    for k in range(n):
        sc = Spacecraft.sunsync(
            altitude_km=ALT_KM,
            node_solar_time=LTDN,
            epoch=T0,
            ma_deg=k * 360.0 / n,
        )
        sensor = RectangularSensor(
            theta1_deg=10.0,
            theta2_deg=40.0,
            attitude_law=FixedAttitudeLaw.nadir(),
        )
        sc.add_sensor(sensor)
        sats.append(sc)
    return sats


def main() -> None:
    constellation = build_constellation(3)

    viewer = CesiumViewer(title="SSO Constellation — US Coverage")

    for i, sc in enumerate(constellation):
        viewer.add_spacecraft(
            sc,
            T0,
            T1,
            STEP,
            color=COLORS[i],
            label=f"SSO-{i + 1}",
	    show_sensors=True
        )

    viewer.add_ground_station(GS_WHITE_SANDS, label="White Sands")
    viewer.add_aoi(AOI_US, color=(255, 200, 50, 50), label="CONUS")

    viewer.show()


if __name__ == "__main__":
    main()
