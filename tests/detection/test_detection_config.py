"""Tests for the Phase 6C DetectionConfig additions."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DetectionConfig, AppConfig


def test_backend_defaults_torch():
    assert DetectionConfig().backend == "rfdetr_torch"


def test_trt_backend_and_fields():
    d = DetectionConfig(backend="rfdetr_trt", engine_path="e.engine", precision="int8")
    assert d.backend == "rfdetr_trt"
    assert d.engine_path == "e.engine"
    assert d.precision == "int8"


def test_precision_default_fp16():
    assert DetectionConfig().precision == "fp16"


def test_bad_backend_rejected():
    with pytest.raises(ValidationError):
        DetectionConfig(backend="onnxruntime")  # type: ignore[arg-type]


def test_bad_precision_rejected():
    with pytest.raises(ValidationError):
        DetectionConfig(precision="fp8")  # type: ignore[arg-type]


def test_backward_compatible_in_appconfig():
    app = AppConfig(detection={"model": "nano"})
    assert app.detection.backend == "rfdetr_torch"
    assert app.detection.engine_path == "" and app.detection.precision == "fp16"
