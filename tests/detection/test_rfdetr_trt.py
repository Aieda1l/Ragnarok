"""Tests for the TensorRT engine detector against a fake session (no GPU/engine)."""
from __future__ import annotations
import numpy as np
import pytest
from ragnarok.config.schema import DetectionConfig
from ragnarok.core.types import Frame
from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector
from ragnarok.detection.factory import create_detector


class _FakeSession:
    def __init__(self):
        self.threshold = None
    def infer(self, image, *, threshold):
        self.threshold = threshold
        return ([(10.0, 10.0, 20.0, 30.0)], [0.95], [0])   # boxes, scores, classes


def _frame():
    return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))


def test_detect_maps_session_arrays_to_detections():
    sess = _FakeSession()
    det = RFDETRTensorRTDetector(DetectionConfig(backend="rfdetr_trt", confidence=0.6), session=sess)
    out = det.detect(_frame())
    assert len(out) == 1
    d = list(out)[0]
    assert d.xyxy == (10.0, 10.0, 20.0, 30.0) and d.confidence == 0.95 and d.class_id == 0
    assert sess.threshold == 0.6                  # confidence threaded into the session


def test_detect_empty_session_yields_no_detections():
    class _Empty:
        def infer(self, image, *, threshold):
            return ([], [], [])
    out = RFDETRTensorRTDetector(DetectionConfig(backend="rfdetr_trt"), session=_Empty()).detect(_frame())
    assert len(out) == 0


def test_factory_routes_trt_backend():
    # The factory must select the TRT class for backend=rfdetr_trt. With no
    # injected session the TRT detector calls _build_trt_session, which raises
    # NotImplementedError — and that exception is reachable ONLY via the TRT
    # route, so asserting it uniquely proves the routing (environment-independent;
    # the torch route never raises NotImplementedError).
    cfg = DetectionConfig(backend="rfdetr_trt", engine_path="missing.engine")
    with pytest.raises(NotImplementedError):
        create_detector(cfg)
