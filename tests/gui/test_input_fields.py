from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import set_field, INPUT_FIELDS


def test_input_fields_wellformed_and_target_input_or_arduino():
    assert len(INPUT_FIELDS) >= 3
    for f in INPUT_FIELDS:
        assert f.path.startswith("input.") or f.path.startswith("arduino.")
        assert f.kind in {"text", "int", "choice", "bool"}


def test_input_fields_set_roundtrip_including_text():
    cfg = AppConfig()
    assert set_field(cfg, "input.mouse_driver", "arduino").input.mouse_driver == "arduino"
    assert set_field(cfg, "arduino.port", "COM5").arduino.port == "COM5"
    assert set_field(cfg, "arduino.udp_port", 9000).arduino.udp_port == 9000
    assert set_field(cfg, "arduino.transport", "udp").arduino.transport == "udp"
