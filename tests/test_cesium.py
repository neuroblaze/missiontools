"""Tests for missiontools.cesium._czml CZML packet builders."""

from __future__ import annotations

import json

import numpy as np
import pytest

from missiontools import Spacecraft, GroundStation, AoI
from missiontools import ConicSensor, RectangularSensor, FixedAttitudeLaw


_EPOCH = np.datetime64("2025-01-01T00:00:00", "us")


def _make_sc():
    return Spacecraft(
        a=6_771_000.0,
        e=0.0006,
        i=np.radians(51.6),
        raan=0.0,
        arg_p=0.0,
        ma=0.0,
        epoch=_EPOCH,
    )


# ===========================================================================
# Preamble
# ===========================================================================


class TestPreamble:
    def test_contains_version_and_clock(self):
        from missiontools.cesium._czml import _build_preamble

        t0 = np.datetime64("2025-01-01", "us")
        t1 = np.datetime64("2025-01-02", "us")
        preamble = _build_preamble(t0, t1)
        data = preamble.model_dump(exclude_none=True)
        assert data["id"] == "document"
        assert data["version"] == "1.0"
        assert "clock" in data
        assert data["clock"]["currentTime"] == "2025-01-01T00:00:00Z"


# ===========================================================================
# Spacecraft packets
# ===========================================================================


class TestSpacecraftPackets:
    def test_produces_one_packet(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        packets = build_spacecraft_packets(sc, t0, t1, np.timedelta64(60, "s"))
        assert len(packets) == 1

    def test_packet_has_inertial_position(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        packets = build_spacecraft_packets(sc, t0, t1, np.timedelta64(60, "s"))
        data = packets[0].model_dump(exclude_none=True)
        pos = data["position"]
        assert pos["referenceFrame"] == "INERTIAL"
        assert "epoch" in pos
        assert "cartesian" in pos

    def test_cartesian_values_are_seconds_since_epoch(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(120, "s")
        packets = build_spacecraft_packets(sc, t0, t1, np.timedelta64(60, "s"))
        cartesian = packets[0].model_dump(exclude_none=True)["position"]["cartesian"]
        assert cartesian[0] == 0.0
        assert abs(cartesian[4] - 60.0) < 1e-9
        assert abs(cartesian[8] - 120.0) < 1e-9

    def test_has_point_and_path(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        data = build_spacecraft_packets(sc, t0, t1)[0].model_dump(exclude_none=True)
        assert "point" in data
        assert "path" in data
        assert "label" in data

    def test_custom_id_and_label(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        packets = build_spacecraft_packets(
            sc,
            t0,
            t1,
            np.timedelta64(60, "s"),
            packet_id="my-sc",
            label="ISS",
        )
        data = packets[0].model_dump(exclude_none=True)
        assert data["id"] == "my-sc"
        assert data["name"] == "ISS"
        assert data["label"]["text"] == "ISS"

    def test_empty_propagation_returns_no_packets(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = t0
        packets = build_spacecraft_packets(sc, t0, t1, np.timedelta64(60, "s"))
        assert packets == []

    def test_custom_color(self):
        from missiontools.cesium._czml import build_spacecraft_packets

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        packets = build_spacecraft_packets(
            sc,
            t0,
            t1,
            np.timedelta64(60, "s"),
            color=(255, 0, 0, 255),
        )
        data = packets[0].model_dump(exclude_none=True)
        assert data["point"]["color"]["rgba"] == [255, 0, 0, 255]


# ===========================================================================
# GroundStation packets
# ===========================================================================


class TestGroundStationPacket:
    def test_cartographic_position(self):
        from missiontools.cesium._czml import build_groundstation_packet

        gs = GroundStation(lat=51.5, lon=-0.1)
        packet = build_groundstation_packet(gs)
        data = packet.model_dump(exclude_none=True)
        pos = data["position"]["cartographicDegrees"]
        assert pos[0] == pytest.approx(-0.1)
        assert pos[1] == pytest.approx(51.5)
        assert pos[2] == pytest.approx(0.0)

    def test_has_point(self):
        from missiontools.cesium._czml import build_groundstation_packet

        gs = GroundStation(lat=51.5, lon=-0.1)
        data = build_groundstation_packet(gs).model_dump(exclude_none=True)
        assert "point" in data

    def test_custom_label(self):
        from missiontools.cesium._czml import build_groundstation_packet

        gs = GroundStation(lat=78.2, lon=15.6)
        packet = build_groundstation_packet(gs, label="Svalbard")
        data = packet.model_dump(exclude_none=True)
        assert data["name"] == "Svalbard"
        assert data["label"]["text"] == "Svalbard"

    def test_altitude(self):
        from missiontools.cesium._czml import build_groundstation_packet

        gs = GroundStation(lat=0, lon=0, alt=100.0)
        data = build_groundstation_packet(gs).model_dump(exclude_none=True)
        assert data["position"]["cartographicDegrees"][2] == pytest.approx(100.0)


# ===========================================================================
# AoI packets
# ===========================================================================


class TestAoIPacket:
    def test_geometry_backed_aoi(self):
        from missiontools.cesium._czml import build_aoi_packets

        aoi = AoI.from_region(
            lat_min_deg=30, lat_max_deg=50, lon_min_deg=-10, lon_max_deg=60
        )
        packets = build_aoi_packets(aoi)
        assert len(packets) == 1
        data = packets[0].model_dump(exclude_none=True)
        assert "polygon" in data
        positions = data["polygon"]["positions"]["cartographicDegrees"]
        assert len(positions) >= 12
        assert min(positions[0::3]) == pytest.approx(-10.0)
        assert max(positions[0::3]) == pytest.approx(60.0)
        assert min(positions[1::3]) == pytest.approx(30.0)
        assert max(positions[1::3]) == pytest.approx(50.0)

    def test_plain_aoi_uses_convex_hull(self):
        from missiontools.cesium._czml import build_aoi_packets

        lat = np.array([0, 0, 1, 1], dtype=float)
        lon = np.array([0, 1, 0, 1], dtype=float)
        aoi = AoI(lat, lon)
        packets = build_aoi_packets(aoi)
        data = packets[0].model_dump(exclude_none=True)
        assert "polygon" in data

    def test_custom_color(self):
        from missiontools.cesium._czml import build_aoi_packets

        aoi = AoI.from_region(lat_min_deg=30, lat_max_deg=50)
        packets = build_aoi_packets(aoi, color=(0, 255, 0, 120))
        rgba = packets[0].model_dump(exclude_none=True)["polygon"]["material"][
            "solidColor"
        ]["color"]["rgba"]
        assert rgba == [0, 255, 0, 120]

    def test_multipolygon_produces_multiple_packets(self):
        from missiontools.cesium._czml import build_aoi_packets
        from shapely.geometry import MultiPolygon, box

        mp = MultiPolygon([box(-10, 30, 0, 40), box(50, 30, 60, 40)])
        aoi = AoI.__new__(AoI)
        aoi._geometry = mp
        aoi._lat = None
        aoi._lon = None

        packets = build_aoi_packets(aoi, packet_id="mp", label="Test MP")
        assert len(packets) == 2
        ids = [p.model_dump(exclude_none=True)["id"] for p in packets]
        assert "mp-0" in ids
        assert "mp-1" in ids


# ===========================================================================
# Full document serialization
# ===========================================================================


class TestDocumentSerialization:
    def test_valid_json_output(self):
        from missiontools.cesium._czml import (
            _build_preamble,
            build_spacecraft_packets,
            build_groundstation_packet,
            build_czml_document,
        )

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        gs = GroundStation(lat=51.5, lon=-0.1)

        packets = [
            _build_preamble(t0, t1),
            *build_spacecraft_packets(
                sc, t0, t1, np.timedelta64(60, "s"), packet_id="sc-1"
            ),
            build_groundstation_packet(gs, packet_id="gs-1"),
        ]
        json_str = build_czml_document(packets)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["id"] == "document"
        assert parsed[1]["id"] == "sc-1"
        assert parsed[2]["id"] == "gs-1"


# ===========================================================================
# CesiumViewer integration
# ===========================================================================


class TestCesiumViewer:
    def test_add_spacecraft_and_to_czml(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1, np.timedelta64(60, "s"))
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "document"
        assert parsed[1]["id"] == "sc-1"

    def test_add_ground_station(self):
        from missiontools.cesium import CesiumViewer

        gs = GroundStation(lat=78.2, lon=15.6)
        viewer = CesiumViewer()
        viewer.add_ground_station(gs, label="Svalbard")
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        assert len(parsed) == 2
        assert parsed[1]["id"] == "gs-1"
        assert parsed[1]["name"] == "Svalbard"

    def test_add_aoi(self):
        from missiontools.cesium import CesiumViewer

        aoi = AoI.from_region(lat_min_deg=30, lat_max_deg=50)
        viewer = CesiumViewer()
        viewer.add_aoi(aoi)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        assert len(parsed) == 2
        assert parsed[1]["id"] == "aoi-1"

    def test_mixed_objects(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")
        gs = GroundStation(lat=51.5, lon=-0.1)
        aoi = AoI.from_region(lat_min_deg=30, lat_max_deg=50)

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1, np.timedelta64(60, "s"))
        viewer.add_ground_station(gs)
        viewer.add_aoi(aoi)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        assert len(parsed) == 4
        assert parsed[0]["id"] == "document"
        ids = [p["id"] for p in parsed]
        assert "sc-1" in ids
        assert "gs-1" in ids
        assert "aoi-1" in ids

    def test_multiple_spacecraft(self):
        from missiontools.cesium import CesiumViewer

        sc1 = _make_sc()
        sc2 = Spacecraft(
            a=7_000_000.0,
            e=0.001,
            i=np.radians(97.4),
            raan=0.0,
            arg_p=0.0,
            ma=0.0,
            epoch=_EPOCH,
        )
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc1, t0, t1)
        viewer.add_spacecraft(sc2, t0, t1)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        ids = [p["id"] for p in parsed]
        assert ids.count("document") == 1
        assert "sc-1" in ids
        assert "sc-2" in ids

    def test_preamble_clock_spans_all_objects(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        t0 = np.datetime64("2025-06-01", "us")
        t1 = np.datetime64("2025-06-02", "us")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        clock = parsed[0]["clock"]
        assert clock["currentTime"] == "2025-06-01T00:00:00Z"


# ===========================================================================
# Sensor packet helpers
# ===========================================================================


def _propagate_sc(sc, t0, t1, step_s=60):
    step = np.timedelta64(step_s, "s")
    state = sc.propagate(t0, t1, step)
    return state["r"], state["v"], state["t"]


# ===========================================================================
# Sensor packet generation (unit tests)
# ===========================================================================


class TestBuildSensorPackets:
    def test_no_sensors_returns_empty(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(5400, "s"))
        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        assert packets == []

    def test_conic_sensor_packet_structure(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(5400, "s"))

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        assert len(packets) == 1
        p = packets[0]
        assert p["id"] == "sc-1-sensor-0"
        assert p["position"] == {"reference": "sc-1#position"}
        assert "agi_conicSensor" in p
        assert "orientation" in p
        assert "epoch" in p["orientation"]

        sensor = p["agi_conicSensor"]
        assert sensor["show"] is True
        assert sensor["radius"] == 6_771_000.0
        assert abs(sensor["outerHalfAngle"] - np.radians(10.0)) < 1e-12
        assert abs(sensor["innerHalfAngle"]) < 1e-12
        mat_color = sensor["lateralSurfaceMaterial"]["solidColor"]["color"]["rgba"]
        assert mat_color[:3] == [0, 200, 255]
        assert mat_color[3] == round(255 * 0.2)

    def test_rectangular_sensor_packet_structure(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(
            RectangularSensor(10.0, 40.0, attitude_law=FixedAttitudeLaw.nadir())
        )
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(5400, "s"))

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (255, 100, 50, 255), 6_771_000.0
        )
        assert len(packets) == 1
        p = packets[0]
        assert "agi_rectangularSensor" in p
        assert "agi_conicSensor" not in p

        sensor = p["agi_rectangularSensor"]
        assert abs(sensor["xHalfAngle"] - np.radians(40.0)) < 1e-12
        assert abs(sensor["yHalfAngle"] - np.radians(10.0)) < 1e-12
        assert sensor["radius"] == 6_771_000.0

    def test_multiple_sensors(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(5.0, attitude_law=FixedAttitudeLaw.nadir()))
        sc.add_sensor(
            RectangularSensor(10.0, 20.0, attitude_law=FixedAttitudeLaw.nadir())
        )
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(5400, "s"))

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        assert len(packets) == 2
        assert packets[0]["id"] == "sc-1-sensor-0"
        assert packets[1]["id"] == "sc-1-sensor-1"
        assert "agi_conicSensor" in packets[0]
        assert "agi_rectangularSensor" in packets[1]

    def test_orientation_has_correct_length(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(300, "s"), 60)
        n = len(t)

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        quat_vals = packets[0]["orientation"]["unitQuaternion"]
        assert len(quat_vals) == n * 5

    def test_quaternions_are_unit(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(300, "s"), 60)

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        quat_vals = packets[0]["orientation"]["unitQuaternion"]
        for i in range(0, len(quat_vals), 5):
            qx, qy, qz, qw = quat_vals[i + 1 : i + 5]
            norm = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
            assert abs(norm - 1.0) < 1e-6

    def test_time_tags_are_epoch_relative(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(120, "s"), 60)

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        quat_vals = packets[0]["orientation"]["unitQuaternion"]
        assert quat_vals[0] == pytest.approx(0.0)
        assert abs(quat_vals[5] - 60.0) < 1e-9
        assert abs(quat_vals[10] - 120.0) < 1e-9

    def test_body_mounted_conic_sensor(self):
        from missiontools.cesium._czml import build_sensor_packets

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, body_vector=[0, 0, 1]))
        r, v, t = _propagate_sc(sc, _EPOCH, _EPOCH + np.timedelta64(5400, "s"))

        packets = build_sensor_packets(
            sc, r, v, t, t[0], "sc-1", (0, 200, 255, 255), 6_771_000.0
        )
        assert len(packets) == 1
        assert "agi_conicSensor" in packets[0]


class TestBuildCzmlDocumentWithSensors:
    def test_extra_packets_appended(self):
        from missiontools.cesium._czml import (
            _build_preamble,
            build_spacecraft_packets,
            build_czml_document,
        )

        sc = _make_sc()
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")
        czml_packets = [
            _build_preamble(t0, t1),
            *build_spacecraft_packets(
                sc, t0, t1, np.timedelta64(60, "s"), packet_id="sc-1"
            ),
        ]
        sensor_dict = {
            "id": "sc-1-sensor-0",
            "agi_conicSensor": {"show": True},
        }
        json_str = build_czml_document(czml_packets, extra_packets=[sensor_dict])
        parsed = json.loads(json_str)
        assert len(parsed) == 3
        assert parsed[2]["id"] == "sc-1-sensor-0"
        assert "agi_conicSensor" in parsed[2]

    def test_no_extra_packets_is_unchanged(self):
        from missiontools.cesium._czml import (
            _build_preamble,
            build_spacecraft_packets,
            build_czml_document,
        )

        sc = _make_sc()
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")
        packets = [
            _build_preamble(t0, t1),
            *build_spacecraft_packets(sc, t0, t1, packet_id="sc-1"),
        ]
        json_str = build_czml_document(packets)
        parsed = json.loads(json_str)
        assert len(parsed) == 2


class TestCesiumViewerSensors:
    def test_show_sensors_false_no_sensor_packets(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1, np.timedelta64(60, "s"))
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        ids = [p["id"] for p in parsed]
        assert "sc-1-sensor-0" not in ids

    def test_show_sensors_true_produces_sensor_packets(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1, np.timedelta64(60, "s"), show_sensors=True)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        ids = [p["id"] for p in parsed]
        assert "sc-1-sensor-0" in ids
        sensor_pkt = [p for p in parsed if p["id"] == "sc-1-sensor-0"][0]
        assert "agi_conicSensor" in sensor_pkt

    def test_default_sensor_length_is_perigee(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(sc, t0, t1, np.timedelta64(60, "s"), show_sensors=True)
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        sensor_pkt = [p for p in parsed if "sensor" in p["id"]][0]
        expected = sc.a * (1.0 - sc.e)
        assert sensor_pkt["agi_conicSensor"]["radius"] == pytest.approx(expected)

    def test_custom_sensor_length(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(
            sc,
            t0,
            t1,
            np.timedelta64(60, "s"),
            show_sensors=True,
            sensor_length=500_000.0,
        )
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        sensor_pkt = [p for p in parsed if "sensor" in p["id"]][0]
        assert sensor_pkt["agi_conicSensor"]["radius"] == 500_000.0

    def test_sensor_color_inherits_spacecraft_color(self):
        from missiontools.cesium import CesiumViewer

        sc = _make_sc()
        sc.add_sensor(ConicSensor(10.0, attitude_law=FixedAttitudeLaw.nadir()))
        t0, t1 = _EPOCH, _EPOCH + np.timedelta64(5400, "s")

        viewer = CesiumViewer()
        viewer.add_spacecraft(
            sc,
            t0,
            t1,
            np.timedelta64(60, "s"),
            color=(255, 0, 0, 255),
            show_sensors=True,
        )
        czml = viewer.to_czml()
        parsed = json.loads(czml)
        sensor_pkt = [p for p in parsed if "sensor" in p["id"]][0]
        mat_color = sensor_pkt["agi_conicSensor"]["lateralSurfaceMaterial"][
            "solidColor"
        ]["color"]["rgba"]
        assert mat_color[0] == 255
        assert mat_color[1] == 0
        assert mat_color[2] == 0
        assert mat_color[3] == round(255 * 0.2)
