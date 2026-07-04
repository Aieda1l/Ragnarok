from PySide6.QtGui import QImage
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.gui.dashboard_panel import DashboardPanel


def _snap(seq, fps=120.0):
    return TelemetrySnapshot(fps=fps, loop_ms_p50=5.0, loop_ms_p99=9.0,
                             detection_count=0, preview=None, seq=seq)


def test_tick_ingests_and_dedups_by_seq(qtbot):
    pub = SnapshotPublisher()
    panel = DashboardPanel(pub)
    qtbot.addWidget(panel)
    pub.publish(_snap(1, fps=100.0))
    panel._tick()
    panel._tick()                                       # same seq -> not double-counted
    assert len(panel.history) == 1
    pub.publish(_snap(2, fps=60.0))
    panel._tick()
    assert panel.history.series("fps") == (100.0, 60.0)
    assert "60" in panel.fps_label.text()


def test_tick_no_snapshot_is_noop(qtbot):
    panel = DashboardPanel(SnapshotPublisher())
    qtbot.addWidget(panel)
    panel._tick()                                       # latest() is None -> no crash
    assert len(panel.history) == 0


def test_paints_without_error(qtbot):
    pub = SnapshotPublisher()
    panel = DashboardPanel(pub)
    qtbot.addWidget(panel)
    panel.resize(400, 200)
    for s in range(1, 6):
        pub.publish(_snap(s, fps=float(100 + s)))
        panel._tick()
    img = QImage(400, 200, QImage.Format_ARGB32)
    img.fill(0)
    panel.render(img)                                   # exercises paintEvent
