from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.recoil_model import parse_pattern_text, format_pattern_text, apply_recoil
from ragnarok.gui.recoil_panel import RecoilPanel


def test_parse_and_format_roundtrip():
    text = "0 2\n0.5 5.0\n1,7"                        # comma + space separators
    pts = parse_pattern_text(text)
    assert pts == ((0.0, 2.0), (0.5, 5.0), (1.0, 7.0))
    assert parse_pattern_text(format_pattern_text(pts)) == pts
    assert parse_pattern_text("garbage\n\n3") == ()   # malformed lines skipped


def test_apply_recoil_swaps_handle():
    h = ConfigHandle(AppConfig())
    new = apply_recoil(h, [(0.0, 2.0), (0.0, 5.0)], scale=1.5, enabled=True)
    assert h.current is new
    assert new.recoil.pattern == ((0.0, 2.0), (0.0, 5.0))
    assert new.recoil.scale == 1.5 and new.recoil.enabled is True


def test_panel_apply_emits_and_swaps(qtbot):
    h = ConfigHandle(AppConfig())
    panel = RecoilPanel(h)
    qtbot.addWidget(panel)
    panel.pattern_edit.setPlainText("0 3\n0 7")
    panel.enabled.setChecked(True)
    panel.scale.setValue(2.0)
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._apply()
    assert h.current.recoil.pattern == ((0.0, 3.0), (0.0, 7.0))
    assert h.current.recoil.enabled is True and h.current.recoil.scale == 2.0
    assert blocker.args[0] is h.current
