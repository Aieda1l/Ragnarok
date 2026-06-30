"""Arduino HID mouse driver over the wire protocol (spec §8).

Frames moves/buttons via aim.protocol and writes them through an injected byte
transport, so the framing + sub-pixel accumulation + int8 HID delta-chunking are
unit-testable with a fake transport. Real serial/UDP transports are lazy and
box-only. SendInput remains the default driver; this is the opt-in device path.
"""
from __future__ import annotations

from ragnarok.aim.mouse import MouseDriver, MouseButton, _FracAccumulator
from ragnarok.aim import protocol as p

_BUTTON_BIT = {MouseButton.LEFT: 1, MouseButton.RIGHT: 2, MouseButton.MIDDLE: 4}


def _button_mask(button: MouseButton) -> int:
    return _BUTTON_BIT[button]


class ArduinoDriver(MouseDriver):
    def __init__(self, *, transport, mode: int = 0, max_step: int = 127) -> None:
        self._t = transport
        self._mode = mode
        self._max = max_step
        self._acc = _FracAccumulator()
        self._mask = 0
        self._connected = False

    def connect(self) -> None:
        if hasattr(self._t, "open"):
            self._t.open()
        self._acc.reset()
        self._mask = 0
        self._connected = True

    def close(self) -> None:
        if hasattr(self._t, "close"):
            self._t.close()
        self._connected = False

    def move_relative(self, dx: float, dy: float) -> None:
        ix, iy = self._acc.step(dx, dy)
        while ix != 0 or iy != 0:
            sx = max(-self._max, min(self._max, ix))
            sy = max(-self._max, min(self._max, iy))
            self._t.write(p.encode_move(sx, sy, buttons=self._mask, mode=self._mode))
            ix -= sx
            iy -= sy

    def set_button(self, button: MouseButton, down: bool) -> None:
        bit = _button_mask(button)
        if down:
            self._mask |= bit
        else:
            self._mask &= ~bit
        self._t.write(p.encode_button(self._mask))
