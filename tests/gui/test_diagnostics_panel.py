from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.diagnostics_panel import DiagnosticsPanel


def test_run_step_populates_metric_labels(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel._run_step()
    text = panel.metrics_label.text()
    assert "Rise" in text and "Overshoot" in text          # formatted metrics shown


def test_run_relay_sets_seeds(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel.widget_for("dead_time_s").setValue(10.0)          # ms -> ensures a limit cycle
    panel.widget_for("lag_tau_s").setValue(20.0)
    panel._run_relay()
    assert panel.last_seeds is not None
    assert "Kp" in panel.seeds_label.text()


def test_run_numeric_sets_seeds(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel.widget_for("n_steps").setValue(120)
    panel._run_numeric()
    assert panel.last_seeds is not None


def test_apply_swaps_handle_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = DiagnosticsPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("dead_time_s").setValue(10.0)
    panel.widget_for("lag_tau_s").setValue(20.0)
    panel._run_relay()
    with qtbot.waitSignal(panel.configChanged, timeout=2000) as blocker:
        panel._apply()
    assert h.current.aim.controller_mode == "pid"
    assert h.current.aim.kp == panel.last_seeds.kp
    assert blocker.args[0] is h.current


def test_apply_without_seeds_is_noop(qtbot):
    h = ConfigHandle(AppConfig())
    panel = DiagnosticsPanel(h)
    qtbot.addWidget(panel)
    before = h.current
    panel._apply()                                          # no run yet -> last_seeds None
    assert h.current is before
