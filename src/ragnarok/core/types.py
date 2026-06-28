from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
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


class Team(str, Enum):
    UNKNOWN = "unknown"
    ENEMY = "enemy"
    TEAMMATE = "teammate"


@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    team: Team = Team.UNKNOWN
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def from_detection(cls, det: "Detection", track_id: int, *, team: "Team" = Team.UNKNOWN,
                       age: int = 0, hits: int = 1, time_since_update: int = 0) -> "Track":
        return cls(track_id=track_id, xyxy=det.xyxy, confidence=det.confidence,
                   class_id=det.class_id, team=team, age=age, hits=hits,
                   time_since_update=time_since_update)


@dataclass(frozen=True)
class Tracks:
    items: tuple[Track, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @classmethod
    def empty(cls) -> "Tracks":
        return cls(items=())
