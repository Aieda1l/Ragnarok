import numpy as np
from types import SimpleNamespace
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector
from ragnarok.core.types import Frame

class _FakeModel:
    def __init__(self): self.threshold = None
    def predict(self, image, threshold=0.5):
        self.threshold = threshold
        return SimpleNamespace(xyxy=np.array([[10.0, 10.0, 20.0, 30.0]]),
                               confidence=np.array([0.95]), class_id=np.array([0]))

def test_detect_returns_detections_and_passes_threshold():
    det = RFDETRTorchDetector(DetectionConfig(confidence=0.6), model=_FakeModel())
    frame = Frame(image=np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))
    out = det.detect(frame)
    assert len(out) == 1
    assert list(out)[0].xyxy == (10.0, 10.0, 20.0, 30.0)
    assert det._model.threshold == 0.6


def test_optimize_fp16_calls_model_with_dtype():
    from ragnarok.detection.rfdetr_torch import _optimize_fp16

    class _Opt:
        def __init__(self):
            self.dtype = "unset"

        def optimize_for_inference(self, dtype=None):
            self.dtype = dtype

    m = _Opt()
    assert _optimize_fp16(m, dtype="FP16") is True  # explicit dtype avoids importing torch
    assert m.dtype == "FP16"


def test_optimize_fp16_is_best_effort_on_failure(recwarn):
    from ragnarok.detection.rfdetr_torch import _optimize_fp16

    class _Bad:
        def optimize_for_inference(self, dtype=None):
            raise RuntimeError("unsupported on this build")

    assert _optimize_fp16(_Bad(), dtype="FP16") is False  # never raises
    assert any("optimize_for_inference" in str(w.message) for w in recwarn.list)


def test_injected_model_is_never_optimized():
    # A model passed in by a caller/test is theirs to manage — the detector must
    # not call optimize_for_inference on it (only on auto-built models).
    class _Tracked:
        def __init__(self):
            self.optimized = False

        def optimize_for_inference(self, dtype=None):
            self.optimized = True

    m = _Tracked()
    RFDETRTorchDetector(DetectionConfig(optimize_fp16=True), model=m)
    assert m.optimized is False
