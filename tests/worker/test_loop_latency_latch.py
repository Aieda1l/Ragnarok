"""Phase 9P — the measured latency is latched until the next measure request
(fixing the one-snapshot lifetime the Calibrate GUI could not reliably catch)."""
from __future__ import annotations

import numpy as np

from ragnarok.worker.loop import WorkerLoop
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.core.types import Frame, Detections


class _Cap:
    def start(self):
        pass

    def stop(self):
        pass

    def grab(self):
        return Frame(image=np.zeros((8, 8, 3), np.uint8), t_capture_ns=0, region=(0, 0, 8, 8))


class _Det:
    def detect(self, frame):
        return Detections.empty()

    def set_confidence(self, c):
        pass


class _Measurer:
    def __init__(self, *a, **k):
        pass

    def run(self):
        return 0.042        # 42 ms


def test_latency_latched_across_ticks(monkeypatch):
    import ragnarok.worker.loop as loopmod
    monkeypatch.setattr(loopmod, "WallLatencyMeasurer", _Measurer)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.set_measure_mouse(object())
    loop.request_latency_measure(0.1)
    loop.tick()                                   # runs the measurement
    assert pub.latest().latency_ms == 42.0
    loop.tick()                                   # a normal tick later
    assert pub.latest().latency_ms == 42.0        # STILL latched (was None before)


def test_latency_cleared_on_new_request(monkeypatch):
    import ragnarok.worker.loop as loopmod
    monkeypatch.setattr(loopmod, "WallLatencyMeasurer", _Measurer)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.set_measure_mouse(object())
    loop.request_latency_measure(0.1)
    loop.tick()
    assert pub.latest().latency_ms == 42.0
    loop.request_latency_measure(0.1)             # new request resets the latch
    loop.tick()
    assert pub.latest().latency_ms == 42.0        # fresh measurement re-latched
