from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.detector_reload_panel import DetectorReloadPanel


class _Loop:
    def __init__(self):
        self.detector = None

    def set_detector(self, d):
        self.detector = d


def test_reload_builds_and_swaps(qtbot):
    loop = _Loop()
    panel = DetectorReloadPanel(ConfigHandle(AppConfig()), loop,
                                build_detector=lambda cfg: "DET")
    qtbot.addWidget(panel)
    panel.reload()
    assert loop.detector == "DET"
    assert "reloaded" in panel.status.text()


def test_reload_failure_is_caught(qtbot):
    def boom(cfg):
        raise RuntimeError("engine missing")
    panel = DetectorReloadPanel(ConfigHandle(AppConfig()), _Loop(), build_detector=boom)
    qtbot.addWidget(panel)
    panel.reload()                               # must not raise
    assert "failed" in panel.status.text()
