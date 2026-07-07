import numpy as np

from ragnarok.config.schema import AppConfig
from ragnarok.gui.overlay_window import FovOverlay
from ragnarok.gui.tuning_model import OVERLAY_FIELDS, set_field
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher


def test_fx_fields_exposed_and_default_off():
    ov = AppConfig().overlay
    assert ov.scanlines is False and ov.chroma is False
    paths = {f.path for f in OVERLAY_FIELDS}
    assert {"overlay.scanlines", "overlay.chroma"} <= paths


def test_fx_toggles_roundtrip():
    c = set_field(AppConfig(), "overlay.scanlines", True)
    assert c.overlay.scanlines is True
    c = set_field(AppConfig(), "overlay.chroma", True)
    assert c.overlay.chroma is True


def test_overlay_paints_with_fx_on_without_crashing(qtbot):
    cfg = AppConfig().model_copy(update={
        "overlay": AppConfig().overlay.model_copy(update={"scanlines": True, "chroma": True})})
    pub = SnapshotPublisher()
    pub.publish(TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                                  detection_count=0, preview=np.zeros((8, 8, 3), np.uint8),
                                  seq=1, roi_region=(0, 0, 100, 100)))
    ov = FovOverlay(pub, lambda: cfg)
    qtbot.addWidget(ov)
    ov.resize(120, 90)
    ov.repaint()                                 # scanline + chroma paint path must not crash
