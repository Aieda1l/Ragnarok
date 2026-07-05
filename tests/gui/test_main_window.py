import numpy as np
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.gui.main_window import MainWindow

def test_window_renders_latest_snapshot(qtbot):
    pub = SnapshotPublisher()
    win = MainWindow(pub)
    qtbot.addWidget(win)
    pub.publish(TelemetrySnapshot(
        fps=123.4, loop_ms_p50=5.0, loop_ms_p99=9.0, detection_count=2,
        preview=np.zeros((100, 100, 3), np.uint8), seq=1))
    win.refresh()  # the QTimer slot, called directly
    assert "123.4" in win.stats_label.text()
    assert "2" in win.stats_label.text()

def test_window_handles_no_snapshot(qtbot):
    win = MainWindow(SnapshotPublisher())
    qtbot.addWidget(win)
    win.refresh()  # must not raise when latest() is None

def test_main_window_embeds_controls_widget(qtbot):
    from PySide6.QtWidgets import QLabel
    panel = QLabel("controls")
    win = MainWindow(SnapshotPublisher(), controls=panel)
    qtbot.addWidget(win)
    assert win.controls is panel
    assert panel.parent() is not None                # actually parented into the window


def test_main_window_is_frameless_with_titlebar(qtbot):
    from PySide6.QtCore import Qt
    win = MainWindow(SnapshotPublisher())
    qtbot.addWidget(win)
    assert win.windowFlags() & Qt.FramelessWindowHint
    assert win.title_bar.title_label.text() == "RAGNAROK"
