"""Phase 9P — tick() reads the detector once, so a GUI-thread set_detector mid-tick
can't leave detect() and observe_lock() on different instances (TOCTOU crash)."""
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


class _PlainDet:
    def detect(self, frame):
        return Detections.empty()

    def set_confidence(self, c):
        pass


class _RoiDet:
    """Detector WITH observe_lock that swaps itself for a plain one mid-detect."""

    def __init__(self, loop):
        self._loop = loop
        self.observed = False

    def detect(self, frame):
        self._loop.set_detector(_PlainDet())   # GUI-thread swap to a detector w/o observe_lock
        return Detections.empty()

    def set_confidence(self, c):
        pass

    def observe_lock(self, center, locked):
        self.observed = True


def test_tick_single_reads_detector_no_crash_and_routes_observe_lock():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), None, StageProfiler(), pub)
    roi = _RoiDet(loop)
    loop.set_detector(roi)
    loop.tick()      # must NOT raise AttributeError
    assert pub.latest() is not None
    # observe_lock must be called on the SAME detector that ran detect() this tick,
    # not skipped because self._det was swapped mid-tick (the TOCTOU the fix closes).
    assert roi.observed is True
