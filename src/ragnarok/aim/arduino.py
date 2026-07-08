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
        # clamp to the int8 HID range (>127 can't be represented per frame) and >=1
        # (max_step=0 would never decrement the chunk loop).
        self._max = max(1, min(127, max_step))
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


class SerialTransport:  # pragma: no cover — box-only (real pyserial)
    """USB-CDC byte transport over pyserial. Lazy import; box-only."""

    def __init__(self, port: str, baud: int) -> None:
        self._port, self._baud, self._ser = port, baud, None

    def open(self) -> None:
        import serial  # lazy: optional box-only dependency
        self._ser = serial.Serial(self._port, self._baud)

    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()


class UdpTransport:  # pragma: no cover — box-only (real socket peer)
    """UDP/WiFi byte transport (stdlib socket). Real I/O is box-only."""

    def __init__(self, host: str, udp_port: int) -> None:
        self._addr = (host, udp_port)
        self._sock = None

    def open(self) -> None:
        import socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def write(self, data: bytes) -> None:
        self._sock.sendto(data, self._addr)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()


_HID_REPORT_LEN = 64          # fixed OUTPUT report size (must match the firmware)
_HID_USAGE_PAGE = 0xFF00      # vendor-defined


class HidTransport:  # pragma: no cover paths marked below — real device I/O is box-only
    """PC->Arduino command channel over a vendor HID OUTPUT report (driverless).

    Carries the same MAKCU frame as the serial/UDP transports, prefixed with HID
    report-id 0x00 and padded to a fixed report length. The real device open uses
    hidapi (lazy import, box-only); tests inject ``self._dev``. This is the
    "both directions over HID" path — no COM port to enumerate.
    """

    def __init__(self, vid: int, pid: int, *, usage_page: int = _HID_USAGE_PAGE,
                 report_len: int = _HID_REPORT_LEN) -> None:
        self._vid, self._pid, self._usage = vid, pid, usage_page
        self._len = report_len
        self._dev = None

    def open(self) -> None:  # pragma: no cover — box-only (real hidapi)
        import hid  # lazy: optional box-only dependency (`pip install hidapi`)
        self._dev = hid.device()
        self._dev.open(self._vid, self._pid)
        self._dev.set_nonblocking(1)

    def write(self, data: bytes) -> None:
        if len(data) > self._len:
            # chunk oversized frames across reports (rare; a MOVE frame is small)
            for i in range(0, len(data), self._len):
                self.write(data[i:i + self._len])
            return
        report = bytes([0x00]) + bytes(data) + bytes(self._len - len(data))
        self._dev.write(report)

    def close(self) -> None:
        if self._dev is not None:
            self._dev.close()


def build_arduino_transport(cfg):
    a = cfg.arduino
    if a.transport == "serial":
        if not a.port:
            raise RuntimeError("arduino.port must be set for the serial transport")
        return SerialTransport(a.port, a.baud)
    if a.transport == "hid":
        if not a.vid or not a.hid_pid:
            raise RuntimeError("arduino.vid and arduino.hid_pid must be set for the hid transport")
        return HidTransport(a.vid, a.hid_pid)
    if not a.host or not a.udp_port:
        raise RuntimeError("arduino.host and arduino.udp_port must be set for the udp transport")
    return UdpTransport(a.host, a.udp_port)
