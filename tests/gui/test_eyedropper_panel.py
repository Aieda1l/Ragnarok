from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.gui.eyedropper_panel import EyedropperPanel


def test_apply_sample_sets_custom_band_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = EyedropperPanel(h, SnapshotPublisher())
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        panel.apply_sample((0, 255, 0))              # sample green
    band = h.current.classification.custom_band
    assert band is not None and len(band) == 6


def test_clear_resets_custom_band(qtbot):
    h = ConfigHandle(AppConfig())
    panel = EyedropperPanel(h, SnapshotPublisher())
    qtbot.addWidget(panel)
    panel.apply_sample((0, 255, 0))
    assert h.current.classification.custom_band is not None
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        panel.clear()
    assert h.current.classification.custom_band is None
