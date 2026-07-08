"""Phase 9P — the capture timestamp comes from an injectable clock (the CI seam
for stamping frame ARRIVAL time instead of consume time; real DXGI is box-only)."""
from __future__ import annotations

import numpy as np

from ragnarok.capture.bettercam_capturer import BetterCamCapturer
from ragnarok.config.schema import CaptureConfig


class _FakeCam:
    def start(self, **k):
        pass

    def stop(self):
        pass

    def get_latest_frame(self):
        return np.zeros((4, 4, 3), np.uint8)


class _FakeMod:
    def create(self, **k):
        return _FakeCam()


def test_grab_uses_injected_clock():
    ticks = iter([1234, 5678])
    cap = BetterCamCapturer(CaptureConfig(), (100, 100),
                            bettercam_module=_FakeMod(), clock=lambda: next(ticks))
    cap.start()
    f = cap.grab()
    assert f.t_capture_ns == 1234
