from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher

def test_publisher_starts_empty():
    pub = SnapshotPublisher()
    assert pub.latest() is None

def test_publish_then_latest_returns_newest():
    pub = SnapshotPublisher()
    s1 = TelemetrySnapshot(fps=100.0, loop_ms_p50=5.0, loop_ms_p99=8.0,
                           detection_count=1, preview=None, seq=1)
    s2 = TelemetrySnapshot(fps=120.0, loop_ms_p50=4.0, loop_ms_p99=7.0,
                           detection_count=2, preview=None, seq=2)
    pub.publish(s1)
    pub.publish(s2)
    assert pub.latest().seq == 2
    assert pub.latest().fps == 120.0

def test_snapshot_tracks_default_empty():
    s = TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                          detection_count=0, preview=None, seq=1)
    assert s.tracks == ()

def test_snapshot_carries_tracks():
    from ragnarok.core.types import Track, Team
    tr = Track(track_id=1, xyxy=(0, 0, 5, 5), confidence=0.9, class_id=0, team=Team.ENEMY)
    s = TelemetrySnapshot(fps=60.0, loop_ms_p50=5.0, loop_ms_p99=8.0,
                          detection_count=1, preview=None, seq=1, tracks=(tr,))
    assert len(s.tracks) == 1 and s.tracks[0].team == Team.ENEMY

def test_snapshot_lock_and_region_default_and_carry():
    s = TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                          detection_count=0, preview=None, seq=1)
    assert s.locked_target_id is None and s.roi_region is None
    s2 = TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                           detection_count=0, preview=None, seq=1,
                           locked_target_id=7, roi_region=(10, 20, 394, 404))
    assert s2.locked_target_id == 7 and s2.roi_region == (10, 20, 394, 404)
