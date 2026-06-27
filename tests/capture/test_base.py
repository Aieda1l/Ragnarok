import numpy as np
from ragnarok.capture.base import centered_region, Capturer
from ragnarok.core.types import Frame

def test_centered_region_math():
    # 384 ROI centered on a 1920x1080 screen
    assert centered_region(384, 1920, 1080) == (768, 348, 1152, 732)

def test_capturer_is_abstract():
    assert hasattr(Capturer, "grab")

class _FakeCapturer(Capturer):
    def start(self): self.started = True
    def grab(self):
        return Frame(image=np.zeros((4, 4, 3), np.uint8), t_capture_ns=1, region=(0, 0, 4, 4))
    def stop(self): self.started = False

def test_fake_capturer_grab_returns_frame():
    c = _FakeCapturer()
    c.start()
    f = c.grab()
    assert isinstance(f, Frame)
    c.stop()
