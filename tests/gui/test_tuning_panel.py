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


def test_commit_noop_does_not_emit_or_swap(qtbot):
    # editingFinished fires on focus-out without an edit -> must not swap/reload.
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    before = h.current
    with qtbot.assertNotEmitted(panel.configChanged):
        panel._commit("aim.kp")                              # value unchanged
    assert h.current is before


def test_loaded_out_of_gui_range_value_is_not_clamped(qtbot):
    # recoil.scale is schema-unbounded; a loaded 8.0 must display as 8.0, not clamp.
    from ragnarok.gui.tuning_model import RECOIL_FIELDS
    cfg = AppConfig().model_copy(
        update={"recoil": AppConfig().recoil.model_copy(update={"scale": 8.0})})
    panel = TuningPanel(ConfigHandle(cfg), fields=RECOIL_FIELDS)
    qtbot.addWidget(panel)
    assert panel.widget_for("recoil.scale").value() == 8.0


def test_out_of_gui_range_value_not_clamped_on_build_or_refresh(qtbot):
    # aim.ki is schema-unbounded above (GUI cap 5.0); a config with ki=8.0 must
    # display 8.0 (range auto-expanded), both at construction and after refresh.
    over = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"ki": 8.0})})
    h = ConfigHandle(over)
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    assert panel.widget_for("aim.ki").value() == 8.0            # not clamped to 5.0 on build
    h.swap(AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"ki": 12.0})}))
    panel.refresh()
    assert panel.widget_for("aim.ki").value() == 12.0           # not clamped on refresh


def test_refresh_repaints_widgets_without_recommitting(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    # swap in a new config behind the panel's back, then refresh
    new = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(
        update={"kp": 1.23, "enabled": True, "aimer": "hybrid"})})
    h.swap(new)
    with qtbot.assertNotEmitted(panel.configChanged):        # refresh must not re-commit
        panel.refresh()
    assert panel.widget_for("aim.kp").value() == 1.23
    assert panel.widget_for("aim.enabled").isChecked() is True
    assert panel.widget_for("aim.aimer").currentText() == "hybrid"


def test_text_field_edits_and_refreshes(qtbot):
    from PySide6.QtWidgets import QLineEdit
    from ragnarok.gui.tuning_model import FieldSpec
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h, fields=(FieldSpec("arduino.port", "Port", "text"),))
    qtbot.addWidget(panel)
    w = panel.widget_for("arduino.port")
    assert isinstance(w, QLineEdit) and w.text() == ""
    w.setText("COM3")
    panel._commit("arduino.port")
    assert h.current.arduino.port == "COM3"
    # refresh from a swapped config, signal-blocked (no re-commit)
    h.swap(AppConfig().model_copy(update={"arduino": AppConfig().arduino.model_copy(update={"port": "COM7"})}))
    with qtbot.assertNotEmitted(panel.configChanged):
        panel.refresh()
    assert panel.widget_for("arduino.port").text() == "COM7"


def test_save_button_invokes_callback(qtbot):
    h = ConfigHandle(AppConfig())
    saved = {}
    panel = TuningPanel(h, on_save=lambda cfg: saved.setdefault("cfg", cfg))
    qtbot.addWidget(panel)
    panel._save()
    assert saved["cfg"] is h.current
