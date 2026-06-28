"""Tests for HSV color classification utilities (CI-safe, synthetic images)."""
from __future__ import annotations

import numpy as np
import pytest

from ragnarok.classification.color import (
    ColorProfile,
    DEFAULT_ENEMY_PROFILES,
    HSVBand,
    WONG_PROFILES,
    color_match_fraction,
    is_enemy_frame,
    ring_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_ring_image(
    size: int,
    xyxy: tuple[float, float, float, float],
    bgr_color: tuple[int, int, int],
    thickness: int = 6,
) -> np.ndarray:
    """Return a black image with a solid colored ring drawn inside *xyxy*."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    # Fill the ring region directly using ring_mask
    mask = ring_mask((size, size), xyxy, thickness=thickness)
    img[mask] = bgr_color
    return img


# ---------------------------------------------------------------------------
# ring_mask
# ---------------------------------------------------------------------------

class TestRingMask:
    def test_shape_matches_input(self):
        m = ring_mask((80, 100), (10, 10, 90, 70))
        assert m.shape == (80, 100)

    def test_ring_pixels_nonzero(self):
        m = ring_mask((80, 100), (10, 10, 90, 70), thickness=4)
        assert m.sum() > 0

    def test_interior_is_false(self):
        """Centre pixels (well inside the box) must NOT be in the ring."""
        m = ring_mask((100, 100), (10, 10, 90, 90), thickness=4)
        # Check a pixel clearly in the interior
        assert m[50, 50] is np.bool_(False) or not m[50, 50]

    def test_border_pixels_are_true(self):
        """Pixels on the outer edge of the box must be in the ring."""
        m = ring_mask((100, 100), (10, 10, 90, 90), thickness=4)
        # Top border row at y=10
        assert m[10, 50]

    def test_zero_area_box_returns_empty(self):
        m = ring_mask((50, 50), (20, 20, 20, 20), thickness=4)
        assert m.sum() == 0


# ---------------------------------------------------------------------------
# color_match_fraction / is_enemy_frame  — YELLOW
# ---------------------------------------------------------------------------

class TestYellowProfile:
    """BGR yellow = (0, 255, 255) → OpenCV HSV H≈30."""

    XYXY = (5.0, 5.0, 55.0, 55.0)
    SIZE = 60
    THICKNESS = 6

    def _img(self, bgr):
        return _solid_ring_image(self.SIZE, self.XYXY, bgr, thickness=self.THICKNESS)

    def test_yellow_ring_high_fraction(self):
        yellow_bgr = (0, 255, 255)  # BGR yellow
        img = self._img(yellow_bgr)
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        frac = color_match_fraction(img, self.XYXY, profile, thickness=self.THICKNESS)
        assert frac >= 0.5, f"Expected >= 0.5, got {frac:.3f}"

    def test_yellow_ring_is_enemy_frame(self):
        yellow_bgr = (0, 255, 255)
        img = self._img(yellow_bgr)
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        assert is_enemy_frame(img, self.XYXY, profile, frac_threshold=0.18, thickness=self.THICKNESS)

    def test_grey_ring_not_enemy(self):
        """Desaturated grey should NOT match the yellow profile."""
        grey_bgr = (128, 128, 128)
        img = self._img(grey_bgr)
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        frac = color_match_fraction(img, self.XYXY, profile, thickness=self.THICKNESS)
        assert frac < 0.18, f"Expected < 0.18 for grey, got {frac:.3f}"

    def test_black_bg_not_enemy(self):
        """All-black image (no ring color) should give frac=0."""
        img = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        frac = color_match_fraction(img, self.XYXY, profile, thickness=self.THICKNESS)
        assert frac == 0.0


# ---------------------------------------------------------------------------
# RED — two-range wraparound
# ---------------------------------------------------------------------------

class TestRedProfile:
    """BGR red = (0, 0, 255) → OpenCV HSV H≈0, which is in the lo range."""

    XYXY = (5.0, 5.0, 55.0, 55.0)
    SIZE = 60
    THICKNESS = 6

    def _img(self, bgr):
        return _solid_ring_image(self.SIZE, self.XYXY, bgr, thickness=self.THICKNESS)

    def test_red_lo_range_detected(self):
        """Pure red (H≈0) hits the lower range (0–10)."""
        red_bgr = (0, 0, 255)
        img = self._img(red_bgr)
        profile = DEFAULT_ENEMY_PROFILES["red"]
        frac = color_match_fraction(img, self.XYXY, profile, thickness=self.THICKNESS)
        assert frac >= 0.5, f"Expected >= 0.5 for red, got {frac:.3f}"

    def test_red_is_enemy_frame(self):
        red_bgr = (0, 0, 255)
        img = self._img(red_bgr)
        profile = DEFAULT_ENEMY_PROFILES["red"]
        assert is_enemy_frame(img, self.XYXY, profile, frac_threshold=0.18, thickness=self.THICKNESS)

    def test_profile_has_two_bands(self):
        """RED must define two bands to handle hue wraparound."""
        profile = DEFAULT_ENEMY_PROFILES["red"]
        assert len(profile.bands) == 2


# ---------------------------------------------------------------------------
# PURPLE
# ---------------------------------------------------------------------------

class TestPurpleProfile:
    """BGR magenta/purple ≈ (180, 0, 180) → OpenCV HSV H≈150."""

    XYXY = (5.0, 5.0, 55.0, 55.0)
    SIZE = 60
    THICKNESS = 6

    def _img(self, bgr):
        return _solid_ring_image(self.SIZE, self.XYXY, bgr, thickness=self.THICKNESS)

    def test_purple_ring_detected(self):
        purple_bgr = (180, 0, 180)  # H≈150 in OpenCV HSV
        img = self._img(purple_bgr)
        profile = DEFAULT_ENEMY_PROFILES["purple"]
        frac = color_match_fraction(img, self.XYXY, profile, thickness=self.THICKNESS)
        assert frac >= 0.5, f"Expected >= 0.5 for purple, got {frac:.3f}"

    def test_purple_is_enemy_frame(self):
        purple_bgr = (180, 0, 180)
        img = self._img(purple_bgr)
        profile = DEFAULT_ENEMY_PROFILES["purple"]
        assert is_enemy_frame(img, self.XYXY, profile, frac_threshold=0.18, thickness=self.THICKNESS)


# ---------------------------------------------------------------------------
# Profiles exist
# ---------------------------------------------------------------------------

class TestProfileDicts:
    def test_default_enemy_profiles_keys(self):
        assert set(DEFAULT_ENEMY_PROFILES.keys()) == {"red", "purple", "yellow"}

    def test_wong_profiles_nonempty(self):
        assert len(WONG_PROFILES) > 0

    def test_all_profiles_have_bands(self):
        for name, prof in {**DEFAULT_ENEMY_PROFILES, **WONG_PROFILES}.items():
            assert len(prof.bands) >= 1, f"Profile '{name}' has no bands"

    def test_hsvband_fields(self):
        band = HSVBand(0, 10, 100, 255, 80, 255)
        assert band.h_lo == 0 and band.h_hi == 10
        assert band.s_lo == 100 and band.s_hi == 255
        assert band.v_lo == 80 and band.v_hi == 255
