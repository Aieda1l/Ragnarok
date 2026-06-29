"""Tests for the pure smart-sampling capture decision."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Detection, Detections
from ragnarok.training.sampling import scene_change_fraction, should_capture


def _img(val):
    return np.full((8, 8, 3), val, np.uint8)


def test_scene_change_identical_is_zero():
    assert scene_change_fraction(_img(100), _img(100)) == 0.0


def test_scene_change_full_swing_is_one():
    assert abs(scene_change_fraction(_img(0), _img(255)) - 1.0) < 1e-6


def test_scene_change_shape_mismatch_is_one():
    assert scene_change_fraction(_img(0), np.zeros((4, 4, 3), np.uint8)) == 1.0


def test_capture_on_low_confidence():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.2, 0),))   # below threshold
    assert should_capture(dets, _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_capture_on_no_detections():
    assert should_capture(Detections.empty(), _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_capture_on_scene_change_even_when_confident():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(0), _img(255),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_no_capture_when_confident_and_static():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is False


def test_capture_when_no_last_saved():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(100), None,
                          conf_threshold=0.5, scene_change_threshold=0.15) is True
