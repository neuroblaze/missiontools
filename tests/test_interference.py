"""Tests for the interference analysis module."""

import numpy as np
import pytest

from missiontools import GroundStation
from missiontools.comm import InterferenceAnalysis
from missiontools.comm.antenna import IsotropicAntenna
from missiontools.condition.condition import AbstractCondition

T0 = np.datetime64("2025-01-01T00:00:00", "us")
T1 = T0 + np.timedelta64(300, "s")


class GapCondition(AbstractCondition):
    """False between gap_start and gap_end (inclusive), True elsewhere."""

    def __init__(self, gap_start, gap_end):
        super().__init__(cache_size=0)
        self._a = gap_start
        self._b = gap_end

    def _compute(self, t):
        return ~((t >= self._a) & (t <= self._b))

    def __repr__(self):
        return f"GapCondition({self._a}, {self._b})"


@pytest.fixture
def elevated_triplet():
    """Three elevated ground stations with clear mutual line of sight.

    With isotropic antennas and 0 dBW/Hz transmit PSD the received
    interferer PSD is roughly -152 dBW/Hz throughout the window, so a
    threshold well below that makes every visible sample an exceedance.
    """
    gs_vtx = GroundStation(lat=0.0, lon=0.0, alt=500e3)
    gs_vrx = GroundStation(lat=0.0, lon=1.0, alt=500e3)
    gs_itx = GroundStation(lat=0.5, lon=0.5, alt=500e3)

    ant_vtx = IsotropicAntenna()
    ant_vrx = IsotropicAntenna()
    ant_itx = IsotropicAntenna()
    gs_vtx.add_antenna(ant_vtx)
    gs_vrx.add_antenna(ant_vrx)
    gs_itx.add_antenna(ant_itx)

    return ant_vtx, ant_vrx, ant_itx


def test_interference_without_conditions(elevated_triplet):
    ant_vtx, ant_vrx, ant_itx = elevated_triplet

    ia = InterferenceAnalysis(f_MHz=8200.0)
    ia.add_victim_tx("VTX", ant_vtx, tx_psd=0.0)
    ia.add_victim_rx("VRX", ant_vrx)
    ia.add_interfering_tx("ITX", ant_itx, tx_psd=0.0)

    events, access_totals = ia.compute(
        psd_threshold=-200.0,
        start_time=T0,
        end_time=T1,
        event_step=10.0,
    )

    assert access_totals["VTX"]["VRX"] == pytest.approx(300.0)
    assert len(events) == 1
    assert events[0]["start_time"] == T0
    assert events[0]["end_time"] == T1

    pct = ia.interference_percentage(
        psd_threshold=-200.0,
        victim_tx="VTX",
        victim_rx="VRX",
        interfering_tx="ITX",
    )
    assert pct == pytest.approx(100.0)


def test_events_do_not_bridge_condition_gaps(elevated_triplet):
    """Regression test: exceedance runs must split where a condition is False.

    The victim transmitter is unavailable for the middle 100 s of a 300 s
    window while the interferer stays above threshold whenever all
    conditions hold.  Events must therefore split at the gap, and the
    interference percentage must be exactly 100 % — previously a single
    gap-spanning event was reported and the percentage exceeded 100 %.
    """
    ant_vtx, ant_vrx, ant_itx = elevated_triplet
    gap_start = T0 + np.timedelta64(100, "s")
    gap_end = T0 + np.timedelta64(200, "s")
    gap = GapCondition(gap_start, gap_end)

    ia = InterferenceAnalysis(f_MHz=8200.0)
    ia.add_victim_tx("VTX", ant_vtx, tx_psd=0.0, condition=gap)
    ia.add_victim_rx("VRX", ant_vrx)
    ia.add_interfering_tx("ITX", ant_itx, tx_psd=0.0)

    events, access_totals = ia.compute(
        psd_threshold=-200.0,
        start_time=T0,
        end_time=T1,
        event_step=10.0,
    )

    # Denominator: condition-True samples are 0-90 s and 210-300 s.
    assert access_totals["VTX"]["VRX"] == pytest.approx(180.0)

    # The gap must split the exceedance into two events that avoid it.
    assert len(events) == 2
    for ev in events:
        assert ev["end_time"] <= gap_start or ev["start_time"] >= gap_end
        assert np.all(gap.at(ev["times"]))
        assert np.all(np.isfinite(ev["interferer_psd"]))
        assert np.all(np.isfinite(ev["victim_psd"]))

    pct = ia.interference_percentage(
        psd_threshold=-200.0,
        victim_tx="VTX",
        victim_rx="VRX",
        interfering_tx="ITX",
    )
    assert pct == pytest.approx(100.0)
    assert pct <= 100.0


def test_rethresholding_does_not_bridge_condition_gaps(elevated_triplet):
    """Same scenario, but re-thresholded above the compute() threshold.

    ``interference_percentage`` re-derives exceedance runs from the cached
    per-event PSD arrays, which must also not span condition-False gaps.
    """
    ant_vtx, ant_vrx, ant_itx = elevated_triplet
    gap = GapCondition(T0 + np.timedelta64(100, "s"), T0 + np.timedelta64(200, "s"))

    ia = InterferenceAnalysis(f_MHz=8200.0)
    ia.add_victim_tx("VTX", ant_vtx, tx_psd=0.0, condition=gap)
    ia.add_victim_rx("VRX", ant_vrx)
    ia.add_interfering_tx("ITX", ant_itx, tx_psd=0.0)

    ia.compute(
        psd_threshold=-200.0,
        start_time=T0,
        end_time=T1,
        event_step=10.0,
    )

    # -190 dBW/Hz is still well below the ~-152 dBW/Hz received PSD, so
    # every condition-True sample remains an exceedance.
    pct = ia.interference_percentage(
        psd_threshold=-190.0,
        victim_tx="VTX",
        victim_rx="VRX",
        interfering_tx="ITX",
    )
    assert pct == pytest.approx(100.0)
    assert pct <= 100.0
