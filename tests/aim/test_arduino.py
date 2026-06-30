"""Tests for ArduinoDriver framing/chunking over a fake transport (no serial/MCU)."""
from __future__ import annotations
import struct
from ragnarok.aim.mouse import MouseButton
from ragnarok.aim.arduino import ArduinoDriver
from ragnarok.aim import protocol as p


class _FakeTransport:
    def __init__(self):
        self.writes: list[bytes] = []
        self.opened = False
        self.closed = False
    def open(self):
        self.opened = True
    def close(self):
        self.closed = True
    def write(self, data):
        self.writes.append(bytes(data))


def _moves(t):
    out = []
    for w in t.writes:
        cmd, payload = p.decode_frame(w)
        if cmd == p.CMD_MOVE:
            dx, dy, buttons, mode = struct.unpack("<hhBB", payload)
            out.append((dx, dy, buttons))
    return out


def test_small_move_one_frame():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect()
    d.move_relative(10.0, -5.0)
    assert _moves(t) == [(10, -5, 0)]
    assert t.opened is True


def test_large_move_chunked_to_127():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect()
    d.move_relative(300.0, 0.0)             # 300 -> 127 + 127 + 46
    chunks = [m[0] for m in _moves(t)]
    assert chunks == [127, 127, 46]
    assert sum(chunks) == 300


def test_subpixel_accumulates_until_whole_pixel():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect()
    d.move_relative(0.4, 0.0)               # < 1 px -> no frame
    assert _moves(t) == []
    d.move_relative(0.7, 0.0)               # 0.4+0.7=1.1 -> emit 1
    assert _moves(t) == [(1, 0, 0)]


def test_set_button_sends_mask_and_move_carries_it():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect()
    d.set_button(MouseButton.LEFT, True)
    cmd, payload = p.decode_frame(t.writes[-1])
    assert cmd == p.CMD_BUTTON and payload == b"\x01"
    d.move_relative(5.0, 0.0)
    assert _moves(t)[-1] == (5, 0, 1)       # move frame carries the held button mask
    d.set_button(MouseButton.LEFT, False)
    assert p.decode_frame(t.writes[-1])[1] == b"\x00"


def test_max_step_clamped_to_int8_range():
    # max_step > 127 must be clamped so MOVE frames stay within the int8 HID range.
    t = _FakeTransport()
    d = ArduinoDriver(transport=t, max_step=200)
    d.connect()
    d.move_relative(300.0, 0.0)
    chunks = [m[0] for m in _moves(t)]
    assert max(chunks) <= 127 and sum(chunks) == 300


def test_close_calls_transport_close():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect(); d.close()
    assert t.closed is True
