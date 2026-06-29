"""Tests for TrainingConfig + its nesting in AppConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import TrainingConfig, AppConfig


def test_defaults():
    t = TrainingConfig()
    assert t.frames_dir == "captures"
    assert t.dataset_dir == "datasets"
    assert t.engines_dir == "engines"
    assert t.roboflow_workspace == "" and t.roboflow_project == ""
    assert t.roboflow_version == 1
    assert t.capture_conf_threshold == 0.5
    assert t.scene_change_threshold == 0.15
    assert t.min_capture_interval_s == 0.5
    assert t.hard_example_conf_threshold == 0.5


def test_no_api_key_field():
    # The Roboflow API key must NOT be a config field (env var only).
    assert "api_key" not in TrainingConfig.model_fields
    assert "roboflow_api_key" not in TrainingConfig.model_fields


def test_bounds():
    with pytest.raises(ValidationError):
        TrainingConfig(capture_conf_threshold=1.5)
    with pytest.raises(ValidationError):
        TrainingConfig(roboflow_version=0)


def test_nested_and_backward_compatible():
    assert isinstance(AppConfig().training, TrainingConfig)
    app = AppConfig(detection={"model": "nano"})       # no [training] section
    assert app.training.frames_dir == "captures"


def test_frozen():
    with pytest.raises(Exception):
        TrainingConfig().frames_dir = "x"
