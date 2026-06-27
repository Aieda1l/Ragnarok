import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop

class _Cap:
    def start(self): ...
    def grab(self):
        return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))
    def stop(self): ...

class _Det:
    def detect(self, frame):
        return Detections(items=(Detection((0, 0, 10, 10), 0.9, 0),))

def test_tick_publishes_snapshot_with_detection_count():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.tick()
    snap = pub.latest()
    assert snap is not None
    assert snap.detection_count == 1
    assert snap.seq == 1
    assert snap.preview is not None  # downscaled frame attached

def test_tick_skips_publish_on_no_frame():
    class _NoCap(_Cap):
        def grab(self): return None
    pub = SnapshotPublisher()
    loop = WorkerLoop(_NoCap(), _Det(), StageProfiler(), pub)
    loop.tick()
    assert pub.latest() is None

def test_seq_increments():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.tick(); loop.tick()
    assert pub.latest().seq == 2
