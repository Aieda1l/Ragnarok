"""Tests for DiagnosticsConfig + its nesting in AppConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DiagnosticsConfig, AppConfig


def test_defaults():
    d = DiagnosticsConfig()
    assert d.step_px == 200.0
    assert d.sample_hz == 1000.0
    assert d.timeout_s == 1.0
    assert d.settle_band_frac == 0.02
    assert d.rise_lo == 0.1 and d.rise_hi == 0.9
    assert d.dead_frac == 0.05
    assert d.reg_max_overshoot_pct == 5.0


def test_bounds():
    with pytest.raises(ValidationError):
        DiagnosticsConfig(step_px=0.0)
    with pytest.raises(ValidationError):
        DiagnosticsConfig(settle_band_frac=1.0)


def test_nested_and_backward_compatible():
    assert isinstance(AppConfig().diagnostics, DiagnosticsConfig)
    app = AppConfig(detection={"model": "nano"})
    assert app.diagnostics.sample_hz == 1000.0


def test_frozen():
    with pytest.raises(Exception):
        DiagnosticsConfig().step_px = 1.0
