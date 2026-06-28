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

def test_tick_publishes_tracks_from_injected_tracker():
    from ragnarok.core.types import Track, Tracks
    class _Trk:
        def update(self, detections, ego_affine=None):
            return Tracks(items=(Track(track_id=42, xyxy=(0, 0, 10, 10),
                                       confidence=0.9, class_id=0),))
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, tracker=_Trk())
    loop.tick()
    snap = pub.latest()
    assert len(snap.tracks) == 1 and snap.tracks[0].track_id == 42

def test_profiler_has_track_and_classify_stages():
    pub = SnapshotPublisher()
    prof = StageProfiler()
    WorkerLoop(_Cap(), _Det(), prof, pub).tick()
    assert "track" in prof.stages() and "classify" in prof.stages()
