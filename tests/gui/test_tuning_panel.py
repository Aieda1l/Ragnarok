from PySide6.QtWidgets import QDoubleSpinBox, QCheckBox, QComboBox
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.tuning_panel import TuningPanel


def test_panel_builds_a_widget_per_field(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    assert isinstance(panel.widget_for("aim.kp"), QDoubleSpinBox)
    assert isinstance(panel.widget_for("aim.enabled"), QCheckBox)
    assert isinstance(panel.widget_for("aim.aimer"), QComboBox)
    # initialised from the handle's current config
    assert panel.widget_for("aim.kp").value() == AppConfig().aim.kp


def test_editing_a_field_swaps_handle_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("aim.kp").setValue(0.9)
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._commit("aim.kp")
    assert h.current.aim.kp == 0.9
    assert blocker.args[0].aim.kp == 0.9


def test_choice_and_bool_commit(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("aim.enabled").setChecked(True)
    panel._commit("aim.enabled")
    assert h.current.aim.enabled is True
    panel.widget_for("aim.aimer").setCurrentText("hybrid")
    panel._commit("aim.aimer")
    assert h.current.aim.aimer == "hybrid"


def test_save_button_invokes_callback(qtbot):
    h = ConfigHandle(AppConfig())
    saved = {}
    panel = TuningPanel(h, on_save=lambda cfg: saved.setdefault("cfg", cfg))
    qtbot.addWidget(panel)
    panel._save()
    assert saved["cfg"] is h.current
