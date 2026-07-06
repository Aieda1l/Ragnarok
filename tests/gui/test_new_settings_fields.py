from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import (
    DETECTION_FIELDS, OVERLAY_FIELDS, KEYBIND_FIELDS, INPUT_FIELDS, set_field)


def test_overlay_config_defaults():
    ov = AppConfig().overlay
    assert ov.show_confidence and ov.show_fov and ov.show_boxes
    assert ov.show_tracking_line and ov.diamond_scale == 1.0


def test_new_field_group_paths():
    assert any(f.path == "detection.confidence" for f in DETECTION_FIELDS)
    assert {f.path for f in KEYBIND_FIELDS} >= {"aim.aim_key", "aim.toggle", "trigger.trigger_key"}
    assert any(f.path == "overlay.show_confidence" for f in OVERLAY_FIELDS)
    assert any(f.path == "input.compensate_ballistics" for f in INPUT_FIELDS)


def test_set_new_fields_roundtrip():
    c = set_field(AppConfig(), "detection.confidence", 0.35)
    assert c.detection.confidence == 0.35
    c = set_field(AppConfig(), "overlay.show_confidence", False)
    assert c.overlay.show_confidence is False
    c = set_field(AppConfig(), "aim.toggle", True)
    assert c.aim.toggle is True
    c = set_field(AppConfig(), "input.compensate_ballistics", True)
    assert c.input.compensate_ballistics is True


def test_driver_accepts_compensate_flag():
    from ragnarok.aim.mouse import SendInputMouseDriver
    sent = []
    m = SendInputMouseDriver(send=lambda dx, dy, f: (sent.append((dx, dy, f)), 1)[1],
                             compensate_ballistics=True)
    m.connect()
    m.move_relative(5, 5)
    assert sent == [(5, 5, 1)]            # injected send path constructs + works
