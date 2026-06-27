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
