from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.counts_panel import CountsCalibratePanel


def test_apply_uses_configurable_turn_degrees(qtbot):
    # Folds the old Wizard's arbitrary-turn ability into Calibrate: a 180° turn.
    h = ConfigHandle(AppConfig())
    panel = CountsCalibratePanel(h)
    qtbot.addWidget(panel)
    panel.degrees.setValue(180.0)
    for _ in range(80):
        panel._on_counts(100, 0)              # 8000 counts over 180°
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        panel._apply_360()
    assert abs(h.current.aim.sensitivity - 180.0 / 8000.0) < 1e-9   # 0.0225 deg/count


def test_default_turn_is_360(qtbot):
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    assert panel.degrees.value() == 360.0
