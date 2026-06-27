from __future__ import annotations
from abc import ABC, abstractmethod
from ragnarok.core.types import Frame, Detection, Detections

def to_detections(sv) -> Detections:
    items = tuple(
        Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)),
                  confidence=float(c), class_id=int(k))
        for (x1, y1, x2, y2), c, k in zip(sv.xyxy, sv.confidence, sv.class_id)
    )
    return Detections(items=items)

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> Detections: ...
