from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from ragnarok.config.schema import AppConfig
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay_window import FovOverlay


def _cfg():
    return AppConfig()


def test_overlay_window_has_click_through_flags(qtbot):
    w = FovOverlay(SnapshotPublisher(), _cfg)
    qtbot.addWidget(w)
    flags = w.windowFlags()
    assert flags & Qt.FramelessWindowHint
    assert flags & Qt.WindowStaysOnTopHint
    assert flags & Qt.Tool
    assert w.testAttribute(Qt.WA_TranslucentBackground)
    assert w.testAttribute(Qt.WA_TransparentForMouseEvents)


def test_overlay_window_paints_scene_without_error(qtbot):
    pub = SnapshotPublisher()
    w = FovOverlay(pub, _cfg)
    qtbot.addWidget(w)
    w.resize(1920, 1080)
    tracks = (Track(track_id=3, xyxy=(180, 150, 204, 234), confidence=0.9,
                    class_id=0, team=Team.ENEMY),)
    pub.publish(TelemetrySnapshot(
        fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0, detection_count=1,
        preview=None, seq=1, tracks=tracks,
        locked_target_id=3, roi_region=(768, 348, 1152, 732)))
    img = QImage(1920, 1080, QImage.Format_ARGB32)
    img.fill(0)
    w.render(img)        # exercises paintEvent -> build_scene -> _draw_scene


def test_overlay_window_no_snapshot_is_noop(qtbot):
    w = FovOverlay(SnapshotPublisher(), _cfg)
    qtbot.addWidget(w)
    img = QImage(64, 64, QImage.Format_ARGB32)
    img.fill(0)
    w.render(img)        # must not raise when latest() is None
