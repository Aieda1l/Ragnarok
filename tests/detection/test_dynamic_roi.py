import numpy as np

from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.detection.base import Detector
from ragnarok.detection.roi import DynamicRoiPlanner, RoiPlan, RoiMode
from ragnarok.detection.dynamic import DynamicRoiDetector, render_roi
from ragnarok.config.schema import DynamicRoiConfig


class _FakeDet(Detector):
    def __init__(self):
        self.box = (10.0, 10.0, 20.0, 20.0)
        self.shapes = []

    def detect(self, frame):
        self.shapes.append(frame.image.shape[:2])
        return Detections(items=(Detection(xyxy=self.box, confidence=0.9, class_id=0),))


def _frame(n=100):
    return Frame(image=np.zeros((n, n, 3), np.uint8), t_capture_ns=0, region=(0, 0, n, n))


def _det(dst=100, track=50):
    cfg = DynamicRoiConfig(track_roi_size=track, model_input_px=dst,
                           max_missed_frames=5, rescan_interval_frames=0)
    base = _FakeDet()
    return DynamicRoiDetector(base, DynamicRoiPlanner(cfg), model_input_px=dst), base


def test_search_mode_maps_back_identity():
    d, base = _det()
    out = list(d.detect(_frame()))                 # no lock -> SEARCH (full frame)
    assert base.shapes[-1] == (100, 100)           # letterboxed to dst
    assert out[0].xyxy == (10.0, 10.0, 20.0, 20.0)  # roi==dst -> identity map-back


def test_track_mode_crops_and_maps_detections_back():
    d, base = _det()
    d.detect(_frame())                             # SEARCH
    d.observe_lock((60.0, 60.0), True)             # lock -> next frame TRACK
    out = list(d.detect(_frame()))
    assert base.shapes[-1] == (100, 100)           # 50px crop upscaled to dst=100
    # crop x0=y0=35 (60-25); r=track/dst=0.5 -> box (10,10,20,20)->(40,40,45,45)
    assert out[0].xyxy == (40.0, 40.0, 45.0, 45.0)


def test_render_roi_track_crop_preserves_content():
    img = np.zeros((100, 100, 3), np.uint8)
    img[40:60, 40:60] = 255
    plan = RoiPlan(mode=RoiMode.TRACK, region=(35, 35, 50, 50), letterboxed=False)
    out = render_roi(img, plan, 100)
    assert out.shape == (100, 100, 3)
    assert out.max() == 255                        # the bright crop survives the upscale
