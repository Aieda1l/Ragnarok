import numpy as np

from ragnarok.aim.latency_measure import WallLatencyMeasurer
from ragnarok.core.types import Frame


class _FakeCap:
    def __init__(self, n):
        self._n = n
        self._i = 0

    def grab(self):
        if self._i >= self._n:
            return None
        self._i += 1
        return Frame(image=np.zeros((16, 16, 3), np.uint8), t_capture_ns=0, region=(0, 0, 16, 16))


class _FakeMouse:
    def __init__(self):
        self.cmds = []

    def move_relative(self, dx, dy):
        self.cmds.append(dx)


def test_measurer_recovers_positive_lag_from_delayed_flow():
    # Orchestration test: the scene flow is the negated command delayed by a few
    # observations, so the measurer must recover a POSITIVE lag. (estimate_lag's
    # exact-value accuracy is covered by tests/aim/test_latency.py.)
    mouse = _FakeMouse()
    lag_obs = 3

    def shift_fn(prev, cur):
        i = len(mouse.cmds)                       # observations recorded so far
        src = mouse.cmds[i - lag_obs] if i >= lag_obs else 0.0
        return (-src, 0.0)                        # scene moves opposite the view, delayed

    t = {"s": 0.0}

    def clock():
        t["s"] += 0.01
        return t["s"]

    m = WallLatencyMeasurer(_FakeCap(80), mouse, duration_s=0.9, amp=40.0,
                            freq_hz=3.0, shift_fn=shift_fn, clock=clock)
    lag = m.run()
    assert lag is not None
    assert 0.0 < lag < 0.9                         # recovered a sane positive delay


def test_measurer_returns_none_on_too_few_frames():
    t = {"s": 0.0}

    def clock():                                 # MUST advance or run() would spin
        t["s"] += 0.01
        return t["s"]

    assert WallLatencyMeasurer(_FakeCap(2), _FakeMouse(), duration_s=0.05,
                               clock=clock).run() is None      # only 1 shift < 10
