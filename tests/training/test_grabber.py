"""Tests for the FrameGrabber (rate-limit + injected writer; no disk)."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.training.grabber import FrameGrabber


def _frame(val, t_ns):
    return Frame(np.full((8, 8, 3), val, np.uint8), t_capture_ns=t_ns, region=(0, 0, 8, 8))


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def _uncertain():
    return Detections.empty()              # no detections -> always "worth capturing"


def test_saves_first_uncertain_frame():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append((img.copy(), t)),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.5, clock=clk)
    assert g.offer(_frame(100, 1), _uncertain()) is True
    assert g.count == 1 and len(saved) == 1 and saved[0][1] == 1


def test_rate_limit_blocks_within_interval():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append(t),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.5, clock=clk)
    g.offer(_frame(100, 1), _uncertain())          # saved at t=0
    clk.t = 200_000_000                            # 200 ms < 500 ms
    assert g.offer(_frame(0, 2), _uncertain()) is False
    clk.t = 600_000_000                            # 600 ms >= 500 ms
    assert g.offer(_frame(0, 3), _uncertain()) is True
    assert g.count == 2


def test_skips_confident_static_frames():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append(t),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.0, clock=clk)
    confident = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    g.offer(_frame(100, 1), confident)             # first -> last_saved None -> saved
    assert g.count == 1
    clk.t = 1_000_000_000
    assert g.offer(_frame(100, 2), confident) is False   # confident + identical -> skip
    assert g.count == 1
