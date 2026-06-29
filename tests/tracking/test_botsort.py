"""CI-safe synthetic tests for the vendored motion-only BoT-SORT wrapper.

No torch / cv2 / weights / display: pure numpy + scipy.
"""
import numpy as np

from ragnarok.core.types import Detection, Detections, Team
from ragnarok.tracking.botsort import BotSortTracker


def _dets(*boxes, conf=0.9, cls=0):
    return Detections(tuple(Detection(b, conf, cls) for b in boxes))


def test_id_persists_for_slowly_moving_box():
    tr = BotSortTracker()
    # A box drifting a few pixels per frame should keep one stable id.
    boxes = [
        (100.0, 100.0, 140.0, 160.0),
        (103.0, 101.0, 143.0, 161.0),
        (106.0, 102.0, 146.0, 162.0),
        (109.0, 103.0, 149.0, 163.0),
    ]
    ids = []
    out = None
    for b in boxes:
        out = tr.update(_dets(b))
        assert len(out) >= 1
        ids.append(out.items[0].track_id)

    # Same id across every frame.
    assert len(set(ids)) == 1
    # A confirmed track has accumulated hits across frames.
    final = out.items[0]
    assert final.hits >= 2
    assert final.team == Team.UNKNOWN
    assert final.time_since_update == 0


def test_two_separated_boxes_get_distinct_ids():
    tr = BotSortTracker()
    a = (100.0, 100.0, 140.0, 160.0)
    b = (400.0, 400.0, 440.0, 460.0)
    out = None
    for _ in range(3):
        out = tr.update(_dets(a, b))
    assert len(out) == 2
    ids = {t.track_id for t in out}
    assert len(ids) == 2


def test_empty_detections_yield_no_tracks():
    tr = BotSortTracker()
    out = tr.update(Detections.empty())
    assert len(out) == 0


def test_track_xyxy_close_to_detection():
    tr = BotSortTracker()
    box = (50.0, 60.0, 90.0, 120.0)
    out = None
    for _ in range(2):
        out = tr.update(_dets(box))
    t = out.items[0]
    assert np.allclose(t.xyxy, box, atol=5.0)


def test_update_forwards_frame_to_ego():
    from ragnarok.tracking.egomotion import EgoMotion

    class _SpyEgo(EgoMotion):
        def __init__(self):
            self.seen = "unset"
        def estimate(self, frame):
            self.seen = frame                     # capture what the core passed
            return np.eye(2, 3, dtype=np.float32)

    spy = _SpyEgo()
    tr = BotSortTracker(ego=spy)
    sentinel = object()
    tr.update(_dets((10, 10, 30, 50)), sentinel)  # 2nd positional arg is `frame`
    assert tr.ego is spy
    assert spy.seen is sentinel                    # the exact frame reached the ego
