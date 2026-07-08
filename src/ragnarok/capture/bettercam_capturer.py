from __future__ import annotations
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.core.types import Frame
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer, centered_region

class BetterCamCapturer(Capturer):
    def __init__(self, config: CaptureConfig, screen_size: tuple[int, int],
                 *, bettercam_module=None, clock=now_ns) -> None:
        self._config = config
        self._region = centered_region(config.roi_size, *screen_size)
        if bettercam_module is None:
            import bettercam  # imported lazily so tests don't need it
            bettercam_module = bettercam
        self._mod = bettercam_module
        self._clock = clock
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
        # BOX-ONLY REFINEMENT: for true ARRIVAL time (removing the 0..1-frame
        # buffer-age jitter that leaks into IMM velocity), source t from the DXGI
        # DXGI_OUTDUPL_FRAME_INFO.LastPresentTime (QPC) the bettercam duplicator
        # exposes, instead of this consume-time clock. The injectable clock is the
        # CI seam; wiring the real DXGI timestamp is verified on the box.
        return Frame(image=np.ascontiguousarray(img), t_capture_ns=self._clock(),
                     region=self._region)

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
