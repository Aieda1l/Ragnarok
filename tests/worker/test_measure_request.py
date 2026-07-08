import numpy as np

from ragnarok.worker.loop import WorkerLoop
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.core.types import Frame, Detections


class _Cap:
    def grab(self):
        return Frame(image=np.zeros((16, 16, 3), np.uint8), t_capture_ns=0, region=(0, 0, 16, 16))

    def stop(self):
        pass


class _Det:
    def detect(self, frame):
        return Detections(items=())


class _Mouse:
    def move_relative(self, dx, dy):
        pass


class _Prof:
    def record(self, *a):
        pass

    def percentiles(self, *a):
        return (0.0, 0.0)


def test_measure_request_runs_and_latches(monkeypatch):
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), _Prof(), pub)
    loop.set_measure_mouse(_Mouse())
    # avoid the real capture loop: force the measurer to return a fixed lag
    monkeypatch.setattr("ragnarok.worker.loop.WallLatencyMeasurer.run", lambda self: 0.037)
    loop.request_latency_measure(duration_s=0.1)

    loop.tick()
    snap = pub.latest()
    assert snap is not None and snap.latency_ms == 37.0        # 0.037 s -> 37.0 ms

    loop.tick()                                                # request already consumed
    # Phase 9P: the result is LATCHED so the 200ms-polling Calibrate GUI can catch it.
    assert pub.latest().latency_ms == 37.0
