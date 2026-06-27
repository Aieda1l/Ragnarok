from __future__ import annotations
import cv2
from ragnarok.core.types import Frame, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector, to_detections

_MODEL_CLASSES = {
    "nano": "RFDETRNano", "small": "RFDETRSmall",
    "medium": "RFDETRMedium", "large": "RFDETRLarge",
}

class RFDETRTorchDetector(Detector):
    def __init__(self, config: DetectionConfig, *, model=None) -> None:
        self._config = config
        if model is None:
            import rfdetr  # lazy: keeps torch/weights out of unit tests
            model = getattr(rfdetr, _MODEL_CLASSES[config.model])()
        self._model = model

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        sv = self._model.predict(rgb, threshold=self._config.confidence)
        return to_detections(sv)
