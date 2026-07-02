from ragnarok.config.schema import AppConfig
from ragnarok.gui.live_config import AimReloader


class _Loop:
    def __init__(self):
        self.controller = "sentinel"
    def set_aim_controller(self, c):
        self.controller = c


def test_reload_builds_and_sets_when_enabled():
    loop = _Loop()
    seen = {}
    def build(cfg, buf):
        seen["cfg"] = cfg
        seen["buf"] = buf
        return "CTRL"
    r = AimReloader(loop, build, commanded_buffer="BUF")
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"enabled": True})})
    r.reload(cfg)
    assert loop.controller == "CTRL"
    assert seen["cfg"] is cfg and seen["buf"] == "BUF"


def test_reload_disables_without_building_when_aim_off():
    loop = _Loop()
    called = {"n": 0}
    def build(cfg, buf):
        called["n"] += 1
        return "CTRL"
    r = AimReloader(loop, build)
    r.reload(AppConfig())                           # aim.enabled defaults False
    assert loop.controller is None
    assert called["n"] == 0                          # no rebuild when disabled
