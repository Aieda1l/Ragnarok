from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Frame:
    image: np.ndarray            # HxWx3 uint8, BGR
    t_capture_ns: int            # now_ns() at grab
    region: tuple[int, int, int, int]  # (left, top, right, bottom) absolute screen coords

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]  # in ROI pixel coords
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

@dataclass(frozen=True)
class Detections:
    items: tuple[Detection, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(items=())
