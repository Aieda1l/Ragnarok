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
