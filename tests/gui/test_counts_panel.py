from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.counts_panel import CountsCalibratePanel


def test_counts_accumulate_and_reset(qtbot):
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel._on_counts(50, 0)
    panel._on_counts(50, -10)
    assert (panel._x, panel._y) == (100, -10)
    panel.reset()
    assert (panel._x, panel._y) == (0, 0)


def test_apply_360_sets_sensitivity_and_deg_per_count_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = CountsCalibratePanel(h)
    qtbot.addWidget(panel)
    for _ in range(160):
        panel._on_counts(100, 0)             # 16000 counts for a 360 turn
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._apply_360()
    assert abs(h.current.aim.sensitivity - 360.0 / 16000.0) < 1e-9      # 0.0225 deg/count
    assert abs(h.current.tracking.deg_per_count - 360.0 / 16000.0) < 1e-9
    assert blocker.args[0] is h.current


def test_apply_360_with_no_counts_is_guarded(qtbot):
    h = ConfigHandle(AppConfig())
    panel = CountsCalibratePanel(h)
    qtbot.addWidget(panel)
    panel._apply_360()                        # no turn recorded
    assert h.current.aim.sensitivity == AppConfig().aim.sensitivity     # unchanged
    assert "360" in panel.result.text()
