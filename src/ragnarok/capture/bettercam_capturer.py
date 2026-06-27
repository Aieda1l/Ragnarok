from __future__ import annotations
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.core.types import Frame
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer, centered_region

class BetterCamCapturer(Capturer):
    def __init__(self, config: CaptureConfig, screen_size: tuple[int, int],
                 *, bettercam_module=None) -> None:
        self._config = config
        self._region = centered_region(config.roi_size, *screen_size)
        if bettercam_module is None:
            import bettercam  # imported lazily so tests don't need it
            bettercam_module = bettercam
        self._mod = bettercam_module
        self._cam = None

    def start(self) -> None:
        self._cam = self._mod.create(output_idx=self._config.monitor_index, output_color="BGR")
        self._cam.start(region=self._region, target_fps=self._config.target_fps, video_mode=False)

    def grab(self) -> Frame | None:
        if self._cam is None:
            return None
        img = self._cam.get_latest_frame()
        if img is None:
            return None
        return Frame(image=np.ascontiguousarray(img), t_capture_ns=now_ns(), region=self._region)

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
