from ragnarok.config.schema import AppConfig, InputConfig
from ragnarok.wiring import build_mouse_driver


def test_build_mouse_driver_selects_sendinput_by_default():
    calls = {"send": 0, "arduino": 0}
    d = build_mouse_driver(
        AppConfig(),
        sendinput_factory=lambda: (calls.__setitem__("send", calls["send"] + 1), "SEND")[1],
        arduino_factory=lambda cfg: (calls.__setitem__("arduino", calls["arduino"] + 1), "ARD")[1])
    assert d == "SEND" and calls == {"send": 1, "arduino": 0}


def test_build_mouse_driver_selects_arduino_and_passes_cfg():
    cfg = AppConfig().model_copy(update={"input": InputConfig(mouse_driver="arduino")})
    seen = {}
    d = build_mouse_driver(
        cfg,
        sendinput_factory=lambda: "SEND",
        arduino_factory=lambda c: (seen.__setitem__("cfg", c), "ARD")[1])
    assert d == "ARD" and seen["cfg"] is cfg
