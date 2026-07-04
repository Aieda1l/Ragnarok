from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.calibration_panel import CalibrationPanel


def test_solve_applies_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = CalibrationPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("counts").setValue(1000.0)
    panel.widget_for("degrees").setValue(22.0)
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._solve()
    assert abs(h.current.tracking.deg_per_count - 0.022) < 1e-9
    assert abs(h.current.aim.sensitivity - 0.022) < 1e-9
    assert blocker.args[0] is h.current
    assert "0.022" in panel.result_label.text()


def test_solve_zero_counts_is_safe_noop(qtbot):
    # solve_deg_per_count raises on zero counts; the panel must not crash / emit
    h = ConfigHandle(AppConfig())
    panel = CalibrationPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("counts").setValue(0.0)
    panel.widget_for("degrees").setValue(22.0)
    before = h.current
    with qtbot.assertNotEmitted(panel.configChanged):
        panel._solve()
    assert h.current is before                       # nothing applied
    assert "invalid" in panel.result_label.text().lower() or "—" in panel.result_label.text()
