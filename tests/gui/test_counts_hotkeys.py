from ragnarok.config.schema import AppConfig, CalibrationConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.aim.keys import FakeKeyProvider, VK
from ragnarok.gui.counts_panel import CountsCalibratePanel


def test_calibration_defaults_and_vk_map():
    cal = CalibrationConfig()
    assert cal.reset_key == "VK_HOME" and cal.apply_key == "VK_END"
    assert VK["VK_HOME"] == 0x24 and VK["VK_END"] == 0x23   # in the VK map for the provider


def test_reset_hotkey_fires_on_rising_edge_only(qtbot):
    rp, ap = FakeKeyProvider(False), FakeKeyProvider(False)
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()), reset_provider=rp, apply_provider=ap)
    qtbot.addWidget(panel)
    panel._on_counts(500, 0)
    rp.down = True
    panel._poll_keys()                     # rising edge -> reset
    assert panel._x == 0
    panel._on_counts(10, 0)
    panel._poll_keys()                     # key still held -> NO re-fire
    assert panel._x == 10
    rp.down = False
    panel._poll_keys()
    rp.down = True
    panel._poll_keys()                     # new rising edge -> reset again
    assert panel._x == 0


def test_apply_hotkey_sets_sensitivity(qtbot):
    h = ConfigHandle(AppConfig())
    rp, ap = FakeKeyProvider(False), FakeKeyProvider(False)
    panel = CountsCalibratePanel(h, reset_provider=rp, apply_provider=ap)
    qtbot.addWidget(panel)
    for _ in range(160):
        panel._on_counts(100, 0)           # 16000 counts / 360
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        ap.down = True
        panel._poll_keys()                 # apply hotkey rising edge
    assert abs(h.current.aim.sensitivity - 360.0 / 16000.0) < 1e-9
