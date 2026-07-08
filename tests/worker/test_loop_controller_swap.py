"""Phase 9P — a controller swapped out via set_aim_controller is released on the
WORKER thread (next tick), not on the swapping thread, so a released trigger button
can't race the worker's in-flight TriggerBot press."""
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


class _Ctl:
    def __init__(self):
        self.released = 0

    def update(self, tracks, t):
        pass

    def release(self):
        self.released += 1


def test_retired_controller_released_on_next_worker_tick():
    old = _Ctl()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), SnapshotPublisher(), aim_controller=old)
    new = _Ctl()
    loop.set_aim_controller(new)
    assert old.released == 0        # NOT released on the swapping (GUI) thread
    loop.tick()                     # worker thread drains + releases the retired controller
    assert old.released == 1
    assert new.released == 0        # the live controller is untouched


def test_disable_retires_and_releases_controller():
    old = _Ctl()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), SnapshotPublisher(), aim_controller=old)
    loop.set_aim_controller(None)   # aim + trigger disabled
    loop.tick()
    assert old.released == 1        # released even though the new controller is None
