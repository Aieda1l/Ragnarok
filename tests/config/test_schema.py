import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig, CaptureConfig, DetectionConfig

def test_defaults():
    cfg = AppConfig()
    assert cfg.capture.roi_size == 384
    assert cfg.detection.model == "small"
    assert cfg.detection.confidence == 0.5

def test_is_frozen():
    cfg = CaptureConfig()
    with pytest.raises(ValidationError):
        cfg.roi_size = 512  # frozen -> ValidationError

def test_rejects_bad_model():
    with pytest.raises(ValidationError):
        DetectionConfig(model="xl")  # not an Apache-2.0 variant
