"""
missiontools.cesium._czml
==========================
Convert missiontools domain objects to CZML packets for CesiumJS rendering.

This module is the pure-Python CZML generation layer. It has no dependency on
pywebview or any rendering backend, making it straightforward to test.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..ground_station import GroundStation
    from ..aoi import AoI
    from ..spacecraft import Spacecraft


def _datetime64_to_iso(t: np.datetime64) -> str:
    """Convert a datetime64[us] to an ISO 8601 string ending in ``Z``."""
    import datetime as _dt

    us = int(
        (t - np.datetime64("1970-01-01T00:00:00", "us"))
        .astype("timedelta64[us]")
        .astype(np.int64)
    )
    return _dt.datetime.fromtimestamp(us / 1e6, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _datetime64_to_epoch_seconds(t: np.datetime64, epoch: np.datetime64) -> float:
    """Seconds from *epoch* to *t*."""
    return float((t - epoch) / np.timedelta64(1, "s"))


def _build_preamble(
    t_start: np.datetime64,
    t_end: np.datetime64,
    name: str = "missiontools",
):
    """Build the CZML document preamble (id='document') with clock settings."""
    from czml3 import Packet
    from czml3.properties import Clock
    from czml3.types import TimeInterval
    from czml3.enums import ClockRanges, ClockSteps

    return Packet(
        id="document",
        version="1.0",
        name=name,
        clock=Clock(
            currentTime=_datetime64_to_iso(t_start),
            multiplier=60,
            range=ClockRanges.LOOP_STOP,
            step=ClockSteps.SYSTEM_CLOCK_MULTIPLIER,
            interval=TimeInterval(
                start=_datetime64_to_iso(t_start),
                end=_datetime64_to_iso(t_end),
            ),
        ),
    )


def build_spacecraft_packets(
    spacecraft: Spacecraft,
    t_start: np.datetime64,
    t_end: np.datetime64,
    step: np.timedelta64 = np.timedelta64(30, "s"),
    *,
    color: tuple[int, int, int, int] = (0, 200, 255, 255),
    path_color: tuple[int, int, int, int] | None = None,
    label: str | None = None,
    packet_id: str | None = None,
):
    """Generate CZML packets for a :class:`~missiontools.Spacecraft`.

    Positions are expressed in ECI (INERTIAL reference frame) using the
    spacecraft propagator output directly.

    Parameters
    ----------
    spacecraft : Spacecraft
    t_start, t_end : np.datetime64
    step : np.timedelta64
    color : tuple[int, int, int, int]
        RGBA for the spacecraft point icon.
    path_color : tuple[int, int, int, int] | None
        RGBA for the orbit trail.  Defaults to *color* with alpha 180.
    label : str | None
        Display name.  Defaults to the packet id.
    packet_id : str | None
        Unique CZML id.  Defaults to ``"sc-<N>"``.

    Returns
    -------
    list[Packet]
    """
    from czml3 import Packet
    from czml3.properties import (
        Point,
        Path,
        Color,
        Label,
        PolylineMaterial,
        SolidColorMaterial,
    )
    from czml3.types import Cartesian2Value, Cartesian3Value, TimeInterval
    from czml3.enums import ReferenceFrames, VerticalOrigins

    state = spacecraft.propagate(t_start, t_end, step)
    t = state["t"]
    r = state["r"]

    if len(t) == 0:
        return []

    epoch = t[0]
    cartesian_values: list[float] = []
    for i in range(len(t)):
        cartesian_values.append(_datetime64_to_epoch_seconds(t[i], epoch))
        cartesian_values.extend(r[i].tolist())

    if path_color is None:
        path_color = (color[0], color[1], color[2], 180)

    pid = packet_id or f"sc-{id(spacecraft) & 0xFFFF:x}"
    display_label = label or pid

    packet = Packet(
        id=pid,
        name=display_label,
        availability=TimeInterval(
            start=_datetime64_to_iso(t[0]),
            end=_datetime64_to_iso(t[-1]),
        ),
        position={
            "referenceFrame": ReferenceFrames.INERTIAL,
            "epoch": _datetime64_to_iso(epoch),
            "cartesian": Cartesian3Value(values=cartesian_values),
        },
        point=Point(
            pixelSize=10,
            color=Color(rgba=list(color)),
        ),
        label=Label(
            text=display_label,
            fillColor=Color(rgba=list(color)),
            verticalOrigin=VerticalOrigins.BOTTOM,
            pixelOffset=Cartesian2Value(values=[0.0, -14.0]),
        ),
        path=Path(
            show=True,
            width=2,
            material=PolylineMaterial(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=list(path_color)),
                ),
            ),
        ),
    )

    return [packet]


def build_groundstation_packet(
    ground_station: GroundStation,
    *,
    color: tuple[int, int, int, int] = (255, 220, 50, 255),
    label: str | None = None,
    packet_id: str | None = None,
):
    """Generate a CZML packet for a :class:`~missiontools.GroundStation`.

    Parameters
    ----------
    ground_station : GroundStation
    color : tuple[int, int, int, int]
        RGBA for the point icon.
    label : str | None
        Display name.
    packet_id : str | None
        Unique CZML id.  Defaults to ``"gs-<N>"``.

    Returns
    -------
    Packet
    """
    from czml3 import Packet
    from czml3.properties import Point, Color, Label
    from czml3.types import Cartesian2Value
    from czml3.enums import VerticalOrigins

    pid = packet_id or f"gs-{id(ground_station) & 0xFFFF:x}"
    display_label = label or f"GS ({ground_station.lat:.1f}, {ground_station.lon:.1f})"

    return Packet(
        id=pid,
        name=display_label,
        position={
            "cartographicDegrees": [
                ground_station.lon,
                ground_station.lat,
                ground_station.alt,
            ]
        },
        point=Point(
            pixelSize=8,
            color=Color(rgba=list(color)),
        ),
        label=Label(
            text=display_label,
            fillColor=Color(rgba=list(color)),
            verticalOrigin=VerticalOrigins.TOP,
            pixelOffset=Cartesian2Value(values=[0.0, 10.0]),
        ),
    )


def _polygon_degrees_from_geom(geom) -> list[float]:
    """Extract cartographic degrees from a Shapely Polygon exterior."""
    degrees: list[float] = []
    for lon, lat in geom.exterior.coords:
        degrees.extend([lon, lat, 0.0])
    return degrees


def build_aoi_packets(
    aoi: AoI,
    *,
    color: tuple[int, int, int, int] = (255, 80, 80, 80),
    label: str | None = None,
    packet_id: str | None = None,
) -> list:
    """Generate CZML packet(s) for an :class:`~missiontools.AoI`.

    If the AoI geometry is a ``MultiPolygon``, one CZML packet is created per
    constituent polygon (each rendered separately on the globe).

    Parameters
    ----------
    aoi : AoI
    color : tuple[int, int, int, int]
        RGBA fill colour (alpha controls transparency).
    label : str | None
        Display name.
    packet_id : str | None
        Base CZML id.  Defaults to ``"aoi-<N>"``.  For ``MultiPolygon``
        geometries, constituent parts are suffixed ``"-0"``, ``"-1"``, etc.

    Returns
    -------
    list[Packet]
    """
    from czml3 import Packet
    from czml3.properties import (
        Polygon,
        PositionList,
        Material,
        SolidColorMaterial,
        Color,
    )

    pid = packet_id or f"aoi-{id(aoi) & 0xFFFF:x}"
    display_label = label or pid

    if aoi.geometry is not None:
        geom = aoi.geometry
        if geom.geom_type == "MultiPolygon":
            polys = geom.geoms
        elif geom.geom_type == "Polygon":
            polys = [geom]
        else:
            polys = [geom.convex_hull]

        packets = []
        for i, poly in enumerate(polys):
            suffix = f"-{i}" if len(polys) > 1 else ""
            degrees = _polygon_degrees_from_geom(poly)
            packets.append(
                Packet(
                    id=f"{pid}{suffix}",
                    name=display_label if len(polys) == 1 else f"{display_label} ({i})",
                    polygon=Polygon(
                        positions=PositionList(cartographicDegrees=degrees),
                        material=Material(
                            solidColor=SolidColorMaterial(color=Color(rgba=list(color)))
                        ),
                    ),
                )
            )
        return packets

    from scipy.spatial import ConvexHull

    lat_deg = aoi.lat
    lon_deg = aoi.lon
    pts = np.column_stack([lon_deg, lat_deg])
    hull = ConvexHull(pts)
    degrees = []
    for idx in hull.vertices:
        degrees.extend([float(lon_deg[idx]), float(lat_deg[idx]), 0.0])
    degrees.extend(
        [float(lon_deg[hull.vertices[0]]), float(lat_deg[hull.vertices[0]]), 0.0]
    )

    return [
        Packet(
            id=pid,
            name=display_label,
            polygon=Polygon(
                positions=PositionList(cartographicDegrees=degrees),
                material=Material(
                    solidColor=SolidColorMaterial(color=Color(rgba=list(color)))
                ),
            ),
        )
    ]


def _eci_to_ecef_rotations(t: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) rotation matrices from ECI to ECEF via GMST."""
    from ..orbit.frames import gmst

    theta = np.atleast_1d(gmst(t)).astype(np.float64)
    c, s = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(theta), np.ones_like(theta)
    return np.array([[c, s, z], [-s, c, z], [z, z, o]]).transpose(2, 0, 1)


def _sensor_frame_eci(
    sensor, r_eci: np.ndarray, v_eci: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Return sensor frame axes in ECI as ``(N, 3, 3)`` matrices.

    Columns are ``[x_axis, y_axis, boresight]`` for each time step.
    """
    from ..sensor.sensor_law import RectangularSensor, _orthonormal_frame

    if isinstance(sensor, RectangularSensor):
        frame = sensor.sensor_frame_eci(r_eci, v_eci, t)
        return frame[np.newaxis] if frame.ndim == 2 else frame

    if sensor._mode == "body":
        law = sensor._spacecraft.attitude_law
        bf = _orthonormal_frame(sensor._body_vector)
        x = np.atleast_2d(law.rotate_from_body(bf[:, 0], r_eci, v_eci, t))
        y = np.atleast_2d(law.rotate_from_body(bf[:, 1], r_eci, v_eci, t))
        z = np.atleast_2d(law.rotate_from_body(bf[:, 2], r_eci, v_eci, t))
    else:
        law = sensor._attitude_law
        x = np.atleast_2d(law.rotate_from_body([1, 0, 0], r_eci, v_eci, t))
        y = np.atleast_2d(law.rotate_from_body([0, 1, 0], r_eci, v_eci, t))
        z = np.atleast_2d(law.rotate_from_body([0, 0, 1], r_eci, v_eci, t))

    return np.stack([x, y, z], axis=-1)


def _sensor_quaternions_ecef(
    sensor,
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    t: np.ndarray,
    epoch: np.datetime64,
) -> list[float]:
    """Compute time-sampled sensor orientation quaternions in ECEF.

    Returns a flat list ``[t0, qx, qy, qz, qw, …]`` with
    epoch-relative seconds for CZML ``unitQuaternion`` values.
    """
    from scipy.spatial.transform import Rotation

    frames_eci = _sensor_frame_eci(sensor, r_eci, v_eci, t)
    Rz = _eci_to_ecef_rotations(t)
    frames_ecef = np.einsum("nij,njk->nik", Rz, frames_eci)
    quats = Rotation.from_matrix(frames_ecef).as_quat()

    epoch_s = ((t - epoch) / np.timedelta64(1, "s")).astype(np.float64)

    values: list[float] = []
    for i in range(len(t)):
        values.append(float(epoch_s[i]))
        values.extend(quats[i].tolist())
    return values


def build_sensor_packets(
    spacecraft: Spacecraft,
    r: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    epoch: np.datetime64,
    packet_id: str,
    color: tuple[int, int, int, int],
    sensor_length: float,
) -> list[dict]:
    """Build sensor visualisation CZML packets as plain dicts.

    Returns plain dicts (not ``czml3.Packet``) because czml3 does not
    support the ``agi_conicSensor`` / ``agi_rectangularSensor`` extensions
    registered by the *cesium-sensor-volumes* plugin.

    Parameters
    ----------
    spacecraft : Spacecraft
    r : ndarray, shape ``(N, 3)``
        ECI positions from propagation.
    v : ndarray, shape ``(N, 3)``
        ECI velocities from propagation.
    t : ndarray, shape ``(N,)``
        datetime64 timestamps.
    epoch : np.datetime64
        Epoch for time-tagged values (typically ``t[0]``).
    packet_id : str
        CZML id of the parent spacecraft packet.
    color : tuple[int, int, int, int]
        Spacecraft RGBA colour (sensor colour inherits with alpha 0.2).
    sensor_length : float
        Sensor volume radius in metres (distance from entity to far end).

    Returns
    -------
    list[dict]
    """
    from ..sensor.sensor_law import ConicSensor, RectangularSensor

    if not spacecraft.sensors:
        return []

    cr, cg, cb, ca = color
    sensor_alpha_255 = round(ca * 0.2)
    intersection_alpha_255 = round(ca * 0.5)

    packets: list[dict] = []
    for idx, sensor in enumerate(spacecraft.sensors):
        quat_vals = _sensor_quaternions_ecef(sensor, r, v, t, epoch)

        sensor_props: dict = {
            "radius": sensor_length,
            "showIntersection": True,
            "intersectionColor": {"rgba": [cr, cg, cb, intersection_alpha_255]},
            "intersectionWidth": 1,
            "lateralSurfaceMaterial": {
                "solidColor": {"color": {"rgba": [cr, cg, cb, sensor_alpha_255]}}
            },
        }

        if isinstance(sensor, ConicSensor):
            sensor_props.update(
                innerHalfAngle=0.0,
                outerHalfAngle=sensor.half_angle_rad,
                minimumClockAngle=0.0,
                maximumClockAngle=2.0 * float(np.pi),
            )
            czml_key = "agi_conicSensor"
        elif isinstance(sensor, RectangularSensor):
            sensor_props.update(
                xHalfAngle=sensor.theta2_rad,
                yHalfAngle=sensor.theta1_rad,
            )
            czml_key = "agi_rectangularSensor"
        else:
            continue

        packet: dict = {
            "id": f"{packet_id}-sensor-{idx}",
            "name": f"{packet_id} sensor {idx}",
            "position": {"reference": f"{packet_id}#position"},
            "orientation": {
                "epoch": _datetime64_to_iso(epoch),
                "unitQuaternion": quat_vals,
            },
            czml_key: sensor_props,
        }

        if sensor.condition is not None:
            intervals = sensor.condition.intervals(t[0], t[-1])
            if intervals:
                parts = [
                    f"{_datetime64_to_iso(start)}/{_datetime64_to_iso(end)}"
                    for start, end in intervals
                ]
                packet["availability"] = ",".join(parts)
            else:
                packet["availability"] = (
                    f"{_datetime64_to_iso(t[0])}/{_datetime64_to_iso(t[0])}"
                )
        else:
            packet["availability"] = (
                f"{_datetime64_to_iso(t[0])}/{_datetime64_to_iso(t[-1])}"
            )

        packets.append(packet)

    return packets


def build_czml_document(
    packets: list,
    extra_packets: list[dict] | None = None,
) -> str:
    """Serialize a list of czml3 packets (including preamble) to JSON.

    Parameters
    ----------
    packets : list
        Must start with the document preamble packet.
    extra_packets : list[dict] | None
        Additional plain-dict packets to append (e.g. sensor volumes).

    Returns
    -------
    str
        JSON string ready for CesiumJS.
    """
    from czml3 import Document

    doc = Document(packets=packets)
    result = doc.model_dump(exclude_none=True)
    if extra_packets:
        result.extend(extra_packets)
    return json.dumps(result)
