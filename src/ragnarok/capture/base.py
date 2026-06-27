from __future__ import annotations
from abc import ABC, abstractmethod
from ragnarok.core.types import Frame

def centered_region(roi_size: int, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
    half = roi_size // 2
    cx, cy = screen_w // 2, screen_h // 2
    return (cx - half, cy - half, cx + half, cy + half)

class Capturer(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def grab(self) -> Frame | None: ...
    @abstractmethod
    def stop(self) -> None: ...
