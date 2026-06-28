# tests/aim/test_mouse_button.py
from ragnarok.aim.mouse import (
    SendInputMouseDriver, MouseButton,
    MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
)


def _recording_driver():
    calls = []
    drv = SendInputMouseDriver(send=lambda dx, dy, flags: calls.append((dx, dy, flags)) or 1)
    drv.connect()
    return drv, calls


def test_left_down_emits_leftdown_flag():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.LEFT, True)
    assert calls == [(0, 0, MOUSEEVENTF_LEFTDOWN)]


def test_left_up_emits_leftup_flag():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.LEFT, False)
    assert calls == [(0, 0, MOUSEEVENTF_LEFTUP)]


def test_right_button_flags():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.RIGHT, True)
    drv.set_button(MouseButton.RIGHT, False)
    assert calls == [(0, 0, MOUSEEVENTF_RIGHTDOWN), (0, 0, MOUSEEVENTF_RIGHTUP)]
