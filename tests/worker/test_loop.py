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
        def update(self, detections, frame=None):
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

def test_aim_controller_update_called_with_tracks():
    class _Aim:
        def __init__(self): self.calls = []
        def update(self, tracks, t_ns): self.calls.append((tuple(tracks), t_ns))
    aim = _Aim()
    pub = SnapshotPublisher()
    WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, aim_controller=aim).tick()
    assert len(aim.calls) == 1
    tracks_seen, t_ns = aim.calls[0]
    assert len(tracks_seen) >= 1 and t_ns == 1   # frame.t_capture_ns from _Cap fake

def test_aim_stage_recorded_when_controller_present():
    class _Aim:
        def update(self, tracks, t_ns): pass
    prof = StageProfiler()
    WorkerLoop(_Cap(), _Det(), prof, SnapshotPublisher(), aim_controller=_Aim()).tick()
    assert "aim" in prof.stages()

def test_loop_publishes_lock_and_region():
    class _CapR(_Cap):
        def grab(self):
            return Frame(np.zeros((384, 384, 3), np.uint8),
                         t_capture_ns=0, region=(100, 50, 484, 434))
    class _Aim:
        target_id = 42
        def update(self, tracks, t_ns): ...
    pub = SnapshotPublisher()
    WorkerLoop(_CapR(), _Det(), StageProfiler(), pub, aim_controller=_Aim()).tick()
    snap = pub.latest()
    assert snap.locked_target_id == 42
    assert snap.roi_region == (100, 50, 484, 434)

def test_loop_lock_id_none_without_controller():
    pub = SnapshotPublisher()
    WorkerLoop(_Cap(), _Det(), StageProfiler(), pub).tick()
    assert pub.latest().locked_target_id is None
    assert pub.latest().roi_region == (0, 0, 384, 384)

def test_set_aim_controller_hotswaps_and_can_disable():
    class _Aim:
        def __init__(self, tid):
            self.target_id = tid
            self.calls = 0
        def update(self, tracks, t_ns):
            self.calls += 1
    a1, a2 = _Aim(1), _Aim(2)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, aim_controller=a1)
    loop.tick()
    assert a1.calls == 1 and pub.latest().locked_target_id == 1
    loop.set_aim_controller(a2)
    loop.tick()
    assert a2.calls == 1 and pub.latest().locked_target_id == 2
    loop.set_aim_controller(None)                  # disable aim live
    loop.tick()
    assert pub.latest().locked_target_id is None
