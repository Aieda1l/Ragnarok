from ragnarok.config.schema import AppConfig, InputConfig


def test_input_defaults_to_sendinput():
    assert AppConfig().input.mouse_driver == "sendinput"


def test_input_accepts_arduino_and_is_frozen():
    cfg = AppConfig().model_copy(update={"input": InputConfig(mouse_driver="arduino")})
    assert cfg.input.mouse_driver == "arduino"
    import pytest
    with pytest.raises(Exception):
        cfg.input.mouse_driver = "sendinput"       # frozen
