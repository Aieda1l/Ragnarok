from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.app import build_tabs
from ragnarok.telemetry.snapshot import SnapshotPublisher


def test_seven_grouped_tabs(qtbot):
    tabs, panels = build_tabs(ConfigHandle(AppConfig()), SnapshotPublisher(),
                              loop=None, on_save=lambda c: None, on_changed=lambda c: None)
    qtbot.addWidget(tabs)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Dashboard", "Aim", "Targeting", "Fire",
                      "Calibrate", "Interface", "Advanced"]
    assert "Profiles" not in titles and "Motion" not in titles   # collapsed / dropped


def test_tuning_panels_returned_for_refresh(qtbot):
    _, panels = build_tabs(ConfigHandle(AppConfig()), SnapshotPublisher(),
                           loop=None, on_save=lambda c: None, on_changed=lambda c: None)
    # Aim + Detection/Tracking/Friend-Foe + Trigger + Keybinds/Overlay/Input + Motion = 9
    assert len(panels) == 9
    assert all(hasattr(p, "refresh") for p in panels)
