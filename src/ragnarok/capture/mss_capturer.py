from __future__ import annotations
import numpy as np
import mss
from ragnarok.core.clock import now_ns
from ragnarok.core.types import Frame
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer, centered_region

class MssCapturer(Capturer):
    def __init__(self, config: CaptureConfig, screen_size: tuple[int, int]) -> None:
        self._region = centered_region(config.roi_size, *screen_size)
        self._sct: mss.mss | None = None

    def start(self) -> None:
        self._sct = mss.mss()

    def grab(self) -> Frame | None:
        if self._sct is None:
            return None
        left, top, right, bottom = self._region
        bbox = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        raw = self._sct.grab(bbox)
        img = np.asarray(raw)[:, :, :3]  # BGRA -> BGR
        return Frame(image=np.ascontiguousarray(img), t_capture_ns=now_ns(), region=self._region)

    def stop(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
