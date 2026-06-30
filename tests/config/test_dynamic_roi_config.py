"""Tests for DynamicRoiConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DynamicRoiConfig, AppConfig


def test_defaults_off():
    d = DynamicRoiConfig()
    assert d.enabled is False
    assert d.track_roi_size == 192 and d.model_input_px == 384
    assert d.max_missed_frames == 5 and d.rescan_interval_frames == 30


def test_bounds():
    with pytest.raises(ValidationError):
        DynamicRoiConfig(max_missed_frames=0)
    with pytest.raises(ValidationError):
        DynamicRoiConfig(track_roi_size=16)


def test_nested_backward_compatible():
    assert isinstance(AppConfig().dynamic_roi, DynamicRoiConfig)
    assert AppConfig(detection={"model": "nano"}).dynamic_roi.enabled is False
