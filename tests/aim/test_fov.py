"""Tests for ragnarok.aim.fov — focal length, FOV→px conversion, aim point.

Task 4 TDD: write tests first, confirm they fail, then implement.
"""
from __future__ import annotations

import math

import pytest

from ragnarok.aim.fov import (
    aim_point,
    crosshair_for_roi,
    dist_to,
    focal_length_px,
    fov_deg_to_radius_px,
)
from ragnarok.core.types import Team, Track


def _enemy(tid: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.ENEMY)


# ---------------------------------------------------------------------------
# focal_length_px
# ---------------------------------------------------------------------------


class TestFocalLengthPx:
    def test_90deg_1920_is_960(self):
        # tan(45°) = 1.0  →  f = (1920/2) / 1.0 = 960
        assert abs(focal_length_px(90.0, 1920) - 960.0) < 1e-9

    def test_formula_arbitrary(self):
        hfov, w = 75.0, 2560
        expected = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
        assert abs(focal_length_px(hfov, w) - expected) < 1e-9

    def test_wider_fov_gives_smaller_f(self):
        # Wider hfov → smaller focal length
        f60 = focal_length_px(60.0, 1920)
        f90 = focal_length_px(90.0, 1920)
        assert f60 > f90


# ---------------------------------------------------------------------------
# fov_deg_to_radius_px
# ---------------------------------------------------------------------------


class TestFovDegToRadiusPx:
    def test_monotonic(self):
        r4 = fov_deg_to_radius_px(4.0, 90.0, 1920)
        r8 = fov_deg_to_radius_px(8.0, 90.0, 1920)
        assert 0.0 < r4 < r8

    def test_8deg_90hfov_1920w(self):
        # radius = f * tan(aim_fov/2) = 960 * tan(4°)
        expected = 960.0 * math.tan(math.radians(4.0))
        assert abs(fov_deg_to_radius_px(8.0, 90.0, 1920) - expected) < 1e-9

    def test_larger_hfov_smaller_radius_same_aim_cone(self):
        # Wider hfov → smaller f → same aim_fov_deg spans fewer pixels
        r_90 = fov_deg_to_radius_px(5.0, 90.0, 1920)
        r_120 = fov_deg_to_radius_px(5.0, 120.0, 1920)
        assert r_90 > r_120

    def test_positive_for_any_valid_input(self):
        assert fov_deg_to_radius_px(1.0, 90.0, 1920) > 0.0


# ---------------------------------------------------------------------------
# crosshair_for_roi
# ---------------------------------------------------------------------------


class TestCrosshairForRoi:
    def test_square_384(self):
        assert crosshair_for_roi(384, 384) == (192.0, 192.0)

    def test_rectangle(self):
        assert crosshair_for_roi(640, 480) == (320.0, 240.0)


# ---------------------------------------------------------------------------
# dist_to
# ---------------------------------------------------------------------------


class TestDistTo:
    def test_345_triangle(self):
        assert abs(dist_to((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-9

    def test_same_point_is_zero(self):
        assert dist_to((5.0, 5.0), (5.0, 5.0)) == 0.0

    def test_negative_direction(self):
        # distance is unsigned
        assert abs(dist_to((10.0, 10.0), (7.0, 6.0)) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# aim_point
# ---------------------------------------------------------------------------


class TestAimPoint:
    def test_head_mode(self):
        tr = _enemy(1, (100.0, 100.0, 200.0, 300.0))
        # cx = (100+200)/2 = 150, ay = 100 + 0.15*200 = 130
        assert aim_point(tr, head_frac=0.15, mode="head") == (150.0, 130.0)

    def test_body_mode(self):
        tr = _enemy(1, (100.0, 100.0, 200.0, 300.0))
        # cx = 150, ay = (100+300)/2 = 200
        assert aim_point(tr, head_frac=0.15, mode="body") == (150.0, 200.0)

    def test_head_frac_half_equals_body(self):
        tr = _enemy(1, (100.0, 100.0, 200.0, 300.0))
        # head_frac=0.5 → ay = 100 + 0.5*200 = 200, same as body center
        p_head05 = aim_point(tr, head_frac=0.5, mode="head")
        p_body = aim_point(tr, head_frac=0.15, mode="body")
        assert p_head05 == p_body

    def test_default_mode_is_head(self):
        tr = _enemy(1, (100.0, 100.0, 200.0, 300.0))
        assert aim_point(tr) == aim_point(tr, mode="head")

    def test_default_head_frac(self):
        tr = _enemy(1, (100.0, 100.0, 200.0, 300.0))
        assert aim_point(tr) == aim_point(tr, head_frac=0.15)
