"""Tests for missiontools.orbit.shadow — cylindrical shadow model."""

import numpy as np
import pytest

from missiontools.orbit import in_sunlight, sun_vec_eci
from missiontools.orbit.constants import EARTH_SEMI_MAJOR_AXIS


# Use a fixed epoch for reproducibility.
EPOCH = np.datetime64('2025-06-21T12:00:00', 'us')


class TestInSunlightScalar:
    """Scalar (single position) tests."""

    def test_sunlit_on_sun_side(self):
        """Spacecraft on the Sun side of Earth is in sunlight."""
        sun = sun_vec_eci(EPOCH)
        r = sun * (EARTH_SEMI_MAJOR_AXIS + 400_000)  # 400 km sunward
        assert in_sunlight(r, EPOCH) == True

    def test_eclipsed_behind_earth(self):
        """Spacecraft directly behind Earth (anti-sun) is eclipsed."""
        sun = sun_vec_eci(EPOCH)
        r = -sun * (EARTH_SEMI_MAJOR_AXIS + 400_000)  # 400 km anti-sunward
        assert in_sunlight(r, EPOCH) == False

    def test_perpendicular_to_sun_line(self):
        """Spacecraft perpendicular to Sun line (well above surface) is lit."""
        sun = sun_vec_eci(EPOCH)
        # Build a vector perpendicular to the Sun direction
        perp = np.cross(sun, [0, 0, 1])
        perp /= np.linalg.norm(perp)
        r = perp * (EARTH_SEMI_MAJOR_AXIS + 400_000)
        assert in_sunlight(r, EPOCH) == True

    def test_behind_earth_but_outside_cylinder(self):
        """Behind Earth but far enough from the shadow cylinder → sunlit."""
        sun = sun_vec_eci(EPOCH)
        perp = np.cross(sun, [0, 0, 1])
        perp /= np.linalg.norm(perp)
        # Behind Earth but displaced perpendicular by more than Earth radius
        r = -sun * (EARTH_SEMI_MAJOR_AXIS + 400_000) + perp * (EARTH_SEMI_MAJOR_AXIS + 100_000)
        assert in_sunlight(r, EPOCH) == True


class TestInSunlightArray:
    """Array (multiple positions) tests."""

    def test_mixed_array(self):
        """Array with one sunlit and one eclipsed position."""
        sun = sun_vec_eci(EPOCH)
        r_lit = sun * (EARTH_SEMI_MAJOR_AXIS + 400_000)
        r_ecl = -sun * (EARTH_SEMI_MAJOR_AXIS + 400_000)
        r = np.array([r_lit, r_ecl])
        t = np.array([EPOCH, EPOCH])

        result = in_sunlight(r, t)
        assert result.shape == (2,)
        assert result[0] == True
        assert result[1] == False

    def test_custom_body_radius(self):
        """Larger body radius puts more positions in shadow."""
        sun = sun_vec_eci(EPOCH)
        perp = np.cross(sun, [0, 0, 1])
        perp /= np.linalg.norm(perp)
        # Position behind Earth, just outside Earth's cylinder
        r = -sun * (EARTH_SEMI_MAJOR_AXIS + 400_000) + perp * (EARTH_SEMI_MAJOR_AXIS + 1000)
        # Lit with normal radius
        assert in_sunlight(r, EPOCH) == True
        # Eclipsed with inflated radius
        assert in_sunlight(r, EPOCH, body_radius=EARTH_SEMI_MAJOR_AXIS * 2) == False
