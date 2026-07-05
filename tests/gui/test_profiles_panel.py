from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.config.profiles import ProfileStore
from ragnarok.gui.profiles_panel import ProfilesPanel


def _panel(tmp_path, cfg=None):
    store = ProfileStore(tmp_path / "profiles")
    handle = ConfigHandle(cfg or AppConfig())
    return ProfilesPanel(store, handle), store, handle


def test_save_as_writes_profile_and_lists_it(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    panel.name_edit.setText("Sniper")
    panel._save_as()
    assert "Sniper" in store.list()
    assert panel.combo.findText("Sniper") >= 0


def test_load_swaps_handle_and_emits(qtbot, tmp_path):
    # a profile saved with kp=0.9, then loaded into a handle holding defaults
    store = ProfileStore(tmp_path / "profiles")
    saved = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"kp": 0.9})})
    store.save("Rifle", saved)
    handle = ConfigHandle(AppConfig())
    panel = ProfilesPanel(store, handle)
    qtbot.addWidget(panel)
    panel.combo.setCurrentText("Rifle")
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._load()
    assert handle.current.aim.kp == 0.9
    assert blocker.args[0].aim.kp == 0.9


def test_delete_removes_profile(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    store.save("Temp", AppConfig())
    panel._refresh_list()
    panel.combo.setCurrentText("Temp")
    panel._delete()
    assert "Temp" not in store.list()


def test_save_as_blank_name_is_noop(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    panel.name_edit.setText("   ")
    panel._save_as()
    assert store.list() == []


def test_export_then_import_path_roundtrips_and_emits(qtbot, tmp_path):
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"kp": 0.7})})
    panel, store, handle = _panel(tmp_path, cfg)
    qtbot.addWidget(panel)
    dst = tmp_path / "export.toml"
    panel.export_path(dst)                         # export live config
    assert dst.exists()

    panel2, _, handle2 = _panel(tmp_path)          # fresh panel with default config
    qtbot.addWidget(panel2)
    with qtbot.waitSignal(panel2.configChanged, timeout=1000) as blocker:
        panel2.import_path(dst)                    # import -> swaps + emits
    assert handle2.current.aim.kp == 0.7
    assert blocker.args[0].aim.kp == 0.7
