import numpy as np
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.bettercam_capturer import BetterCamCapturer
from ragnarok.core.types import Frame

class _FakeCam:
    def __init__(self): self.started_region = None; self.stopped = False
    def start(self, region, target_fps, video_mode=False):
        self.started_region = region; self.target_fps = target_fps
    def get_latest_frame(self):
        return np.zeros((384, 384, 3), dtype=np.uint8)  # BGR
    def stop(self): self.stopped = True

class _FakeModule:
    def __init__(self): self.cam = _FakeCam()
    def create(self, output_idx=0, output_color="BGR"): return self.cam

def test_start_uses_centered_region_and_fps():
    mod = _FakeModule()
    cap = BetterCamCapturer(CaptureConfig(roi_size=384, target_fps=144),
                            screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    assert mod.cam.started_region == (768, 348, 1152, 732)
    assert mod.cam.target_fps == 144

def test_grab_returns_frame_with_timestamp():
    mod = _FakeModule()
    cap = BetterCamCapturer(CaptureConfig(), screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    f = cap.grab()
    assert isinstance(f, Frame)
    assert f.t_capture_ns > 0
    assert f.image.shape == (384, 384, 3)

def test_grab_returns_none_when_no_new_frame():
    mod = _FakeModule()
    mod.cam.get_latest_frame = lambda: None
    cap = BetterCamCapturer(CaptureConfig(), screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    assert cap.grab() is None
