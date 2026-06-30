# Ragnarok Phase 7A — Arduino Protocol & Driver Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CI-safe core of the Arduino HID input path (spec §8): the MAKCU-modeled binary **wire protocol codec** (encode/decode + CRC8), an **`ArduinoDriver`** (implements the existing `MouseDriver` ABC) that frames moves/buttons over an **injected byte transport** with sub-pixel accumulation + HID delta-chunking, and a config + transport factory — so the device path is fully unit-tested without any serial port, socket, MCU, or firmware.

**Architecture:** A pure `aim/protocol.py` builds/parses the `[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]` frames (spec §8.2). `aim/arduino.py`'s `ArduinoDriver` reuses the existing `_FracAccumulator` (sub-pixel) and `MouseButton` from `aim/mouse.py`, chunks deltas to the int8 HID range (±127), and writes frames through an injected `transport.write(bytes)` — CI uses a fake transport that records bytes and decodes them back. The real serial (`pyserial`) / UDP (stdlib socket) transports are thin, lazily-imported, box-only adapters selected by an `ArduinoConfig`. SendInput remains the default driver (spec §8.1); this is the optional hardware-HID + passthrough path.

**Tech Stack:** Python 3.11+, stdlib (`struct`, `socket` for the UDP transport). `pyserial` is an **optional box-only** dependency (lazy). No new test dependencies. Arduino firmware (`.ino`) is box-only and out of scope for this plan.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; the Arduino is the "real-HID-device + physical-mouse-passthrough" input option, not detection evasion (spec §Scope, §8).
- **CI-safe always:** no serial port / socket peer / MCU / `pyserial` / firmware in unit tests. The driver writes through an injected transport; the protocol codec is pure; the real serial/UDP transports are lazy and box-only. Modules import without `pyserial`.
- **SendInput stays the default** (spec §8.1): the R4 native HID caps ~125 Hz, so this Arduino path is opt-in via config, never the default.
- **Wire protocol is exactly** `[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]` with `CMD`: `0x01 MOVE` (`dx:i16, dy:i16, buttons:u8, mode:u8`), `0x02 BUTTON` (`mask:u8`), `0x03 CONFIG`, `0x04 DIAG` (HIL echo). LEN16 and multi-byte fields are **little-endian** (`<`); CRC8 covers `CMD + LEN16 + PAYLOAD` (not the start byte). The PC codec and the (later, box-only) firmware must agree on the CRC8 polynomial — this plan fixes it at **CRC-8 poly 0x07, init 0x00** (no reflect, no xorout).
- **HID delta-chunking:** device-side HID mouse reports are int8 per axis, so each `MOVE` frame's `dx`/`dy` must be within **±127**; larger commanded deltas are split across multiple frames (spec §6.3 "split >127 px for the int8-limited Arduino HID path").
- **No secrets in config** (consistency with the rest): `ArduinoConfig` holds only non-secret port/baud/ip/transport selection.
- **Reuse:** the `MouseDriver` ABC, `MouseButton`, and `_FracAccumulator` already exist in `aim/mouse.py` — consume them, don't reinvent.
- **Frozen pydantic config**, backward-compatible TOML round-trip.
- **TDD, frequent commits, exact file paths.** Match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, lazy heavy imports, module docstrings, injected collaborators).

## Scope Boundary (explicit deferrals)

- **Arduino firmware (`.ino`)** — the RA4M1 HID sketch, USB.cpp polling patch, COM-hiding, VID/PID spoofing, USB Host Shield passthrough (spec §8.3/§8.4) → box-only, separate (non-Python) deliverable. This plan defines the PC-side protocol the firmware must mirror.
- **BLE + ESP-NOW transports** (spec §8.1) → deferred (BLE is config-rate-only; ESP-NOW needs a dongle). The injected-transport seam makes them drop-in later.
- **Real serial / UDP I/O** → box-only. CI tests the frame bytes via a fake transport; the lazy `pyserial`/socket adapters are box-only.
- **HIL latency measurement loop** (spec §11c) → the `DIAG 0x04` *codec* (request + echo-decode) is in scope; the real round-trip timing against an MCU is box-only.
- **Interception driver** (spec §8.1 raw-input fallback) → out of scope (separate from Arduino).

---

## File Structure

**New files:**
- `src/ragnarok/aim/protocol.py` — `crc8`, `encode_move`/`encode_button`/`encode_config`/`encode_diag`, `decode_frame`, `decode_diag_echo`, the `CMD_*`/`MOUSE_*` constants.
- `src/ragnarok/aim/arduino.py` — `ArduinoDriver` (`MouseDriver`) + `_button_mask`; lazy box-only `SerialTransport`/`UdpTransport` + `build_arduino_transport`.
- `tests/aim/test_protocol.py`, `tests/aim/test_arduino.py`, `tests/aim/test_arduino_transport.py`

**Modified files:**
- `src/ragnarok/config/schema.py` — add `ArduinoConfig`, nest in `AppConfig`.
- `tests/config/test_arduino_config.py` (new).

---

## Task 1: Wire protocol codec (frames + CRC8)

**Files:**
- Create: `src/ragnarok/aim/protocol.py`
- Create: `tests/aim/test_protocol.py`

**Interfaces:**
- Consumes: nothing (stdlib `struct`).
- Produces:
  - Constants `START = 0xAA`, `CMD_MOVE = 0x01`, `CMD_BUTTON = 0x02`, `CMD_CONFIG = 0x03`, `CMD_DIAG = 0x04`.
  - `crc8(data: bytes) -> int` — CRC-8 poly 0x07, init 0x00.
  - `encode_move(dx: int, dy: int, *, buttons: int = 0, mode: int = 0) -> bytes` — frames `<hhBB>` (raises `struct.error`/`OverflowError` if dx/dy outside int16; the driver keeps them in range).
  - `encode_button(mask: int) -> bytes`; `encode_config(payload: bytes) -> bytes`; `encode_diag(seq: int = 0) -> bytes`.
  - `decode_frame(frame: bytes) -> tuple[int, bytes]` — returns `(cmd, payload)`; raises `ValueError` on bad start byte, truncated frame, length mismatch, or CRC mismatch.
  - `decode_diag_echo(payload: bytes) -> int` — `<I` microseconds from a DIAG echo payload.

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_protocol.py
"""Tests for the MAKCU-modeled wire protocol codec (spec §8.2)."""
from __future__ import annotations
import struct
import pytest
from ragnarok.aim import protocol as p


def test_crc8_known_vectors():
    assert p.crc8(b"") == 0x00
    assert p.crc8(b"\x01") == 0x07          # poly 0x07, init 0x00


def test_move_frame_roundtrips():
    frame = p.encode_move(100, -50, buttons=1, mode=0)
    assert frame[0] == p.START
    cmd, payload = p.decode_frame(frame)
    assert cmd == p.CMD_MOVE
    dx, dy, buttons, mode = struct.unpack("<hhBB", payload)
    assert (dx, dy, buttons, mode) == (100, -50, 1, 0)


def test_button_and_config_roundtrip():
    cmd, payload = p.decode_frame(p.encode_button(0b101))
    assert cmd == p.CMD_BUTTON and payload == b"\x05"
    cmd, payload = p.decode_frame(p.encode_config(b"\xde\xad"))
    assert cmd == p.CMD_CONFIG and payload == b"\xde\xad"


def test_decode_rejects_bad_start():
    frame = bytearray(p.encode_button(1))
    frame[0] = 0x00
    with pytest.raises(ValueError):
        p.decode_frame(bytes(frame))


def test_decode_rejects_corrupted_crc():
    frame = bytearray(p.encode_move(10, 10))
    frame[-1] ^= 0xFF                       # flip the CRC byte
    with pytest.raises(ValueError):
        p.decode_frame(bytes(frame))


def test_decode_rejects_length_mismatch():
    frame = p.encode_move(10, 10)
    with pytest.raises(ValueError):
        p.decode_frame(frame[:-2])          # truncated payload/crc


def test_diag_echo_decodes_micros():
    cmd, _ = p.decode_frame(p.encode_diag(seq=7))
    assert cmd == p.CMD_DIAG
    assert p.decode_diag_echo(struct.pack("<I", 1234)) == 1234
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_protocol.py -v`
Expected: FAIL — `No module named 'ragnarok.aim.protocol'`.

- [ ] **Step 3: Implement protocol.py**

```python
# src/ragnarok/aim/protocol.py
"""MAKCU-modeled binary wire protocol (spec §8.2).

Frame: [0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]; multi-byte fields little-endian;
CRC8 (poly 0x07, init 0x00) covers CMD+LEN16+PAYLOAD. Shared by the PC drivers
and (later, box-only) the Arduino firmware — both MUST use this exact framing.
"""
from __future__ import annotations

import struct

START = 0xAA
CMD_MOVE = 0x01
CMD_BUTTON = 0x02
CMD_CONFIG = 0x03
CMD_DIAG = 0x04


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _frame(cmd: int, payload: bytes) -> bytes:
    header = bytes([cmd]) + struct.pack("<H", len(payload))
    body = header + payload
    return bytes([START]) + body + bytes([crc8(body)])


def encode_move(dx: int, dy: int, *, buttons: int = 0, mode: int = 0) -> bytes:
    return _frame(CMD_MOVE, struct.pack("<hhBB", dx, dy, buttons, mode))


def encode_button(mask: int) -> bytes:
    return _frame(CMD_BUTTON, struct.pack("<B", mask))


def encode_config(payload: bytes) -> bytes:
    return _frame(CMD_CONFIG, payload)


def encode_diag(seq: int = 0) -> bytes:
    return _frame(CMD_DIAG, struct.pack("<B", seq))


def decode_frame(frame: bytes) -> tuple[int, bytes]:
    if len(frame) < 5:
        raise ValueError("frame too short")
    if frame[0] != START:
        raise ValueError(f"bad start byte {frame[0]:#x}")
    cmd = frame[1]
    (length,) = struct.unpack("<H", frame[2:4])
    end = 4 + length
    if len(frame) != end + 1:
        raise ValueError("frame length mismatch")
    payload = frame[4:end]
    if crc8(frame[1:end]) != frame[end]:
        raise ValueError("crc mismatch")
    return cmd, payload


def decode_diag_echo(payload: bytes) -> int:
    (micros,) = struct.unpack("<I", payload)
    return micros
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_protocol.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/protocol.py tests/aim/test_protocol.py
git commit -m "feat(aim): MAKCU wire protocol codec (frames + CRC8)"
```

---

## Task 2: ArduinoDriver (MouseDriver over an injected transport)

**Files:**
- Create: `src/ragnarok/aim/arduino.py`
- Create: `tests/aim/test_arduino.py`

**Interfaces:**
- Consumes: `MouseDriver`, `MouseButton`, `_FracAccumulator` (`ragnarok.aim.mouse`); `encode_move`/`encode_button`, `decode_frame` (T1).
- Produces:
  - `_button_mask(button: MouseButton) -> int` — `LEFT=1, RIGHT=2, MIDDLE=4`.
  - `ArduinoDriver(*, transport, mode: int = 0, max_step: int = 127)` implementing `MouseDriver`:
    - `connect()` — calls `transport.open()` if present, zeroes the accumulator + button mask, sets connected.
    - `close()` — calls `transport.close()` if present, clears connected.
    - `move_relative(dx, dy)` — sub-pixel accumulate via `_FracAccumulator`; split the integer delta into chunks of ≤ `max_step` (±127) per axis and write an `encode_move(sx, sy, buttons=mask, mode=mode)` frame per chunk via `transport.write(bytes)`. No write when the accumulated integer delta is `(0, 0)`.
    - `set_button(button, down)` — set/clear the bit in the running mask, write an `encode_button(mask)` frame.
  - `transport` is any object with `write(data: bytes) -> None` (and optional `open()`/`close()`); CI uses a fake recorder.

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_arduino.py
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


def test_close_calls_transport_close():
    t = _FakeTransport()
    d = ArduinoDriver(transport=t)
    d.connect(); d.close()
    assert t.closed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_arduino.py -v`
Expected: FAIL — `No module named 'ragnarok.aim.arduino'`.

- [ ] **Step 3: Implement arduino.py**

```python
# src/ragnarok/aim/arduino.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_arduino.py -v`
Expected: PASS

- [ ] **Step 5: Run the aim suite (no regression) + commit**

Run: `python -m pytest tests/aim -q`
Expected: PASS

```bash
git add src/ragnarok/aim/arduino.py tests/aim/test_arduino.py
git commit -m "feat(aim): ArduinoDriver (framed moves/buttons, sub-pixel + int8 chunking, injected transport)"
```

---

## Task 3: ArduinoConfig + transport factory (box-only adapters)

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Modify: `src/ragnarok/aim/arduino.py`
- Test: `tests/config/test_arduino_config.py`, `tests/aim/test_arduino_transport.py`

**Interfaces:**
- Consumes: `AppConfig` pattern.
- Produces:
  - `ArduinoConfig` (frozen): `transport: Literal["serial", "udp"] = "serial"`, `port: str = ""` (COM/tty for serial), `baud: int = Field(default=115200, ge=1200)` (irrelevant on R4 native USB but kept for non-native bridges), `host: str = ""`, `udp_port: int = Field(default=0, ge=0, le=65535)`. Nested as `AppConfig.arduino`. (No secrets.)
  - In `aim/arduino.py`: `SerialTransport(port, baud)` (lazy `import serial` — box-only; `.open/.write/.close`), `UdpTransport(host, udp_port)` (stdlib `socket`, datagram — real but only used at runtime; `.open/.write/.close`), and `build_arduino_transport(cfg)` selecting by `cfg.arduino.transport`, validating the needed fields (raises `RuntimeError` if `serial` without a `port`, or `udp` without `host`/`udp_port`) BEFORE constructing — so the guard paths are unit-testable without `pyserial`/a socket peer.

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_arduino_config.py
"""Tests for ArduinoConfig + nesting."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import ArduinoConfig, AppConfig


def test_defaults():
    a = ArduinoConfig()
    assert a.transport == "serial"
    assert a.port == "" and a.baud == 115200
    assert a.host == "" and a.udp_port == 0


def test_udp_fields_and_validation():
    a = ArduinoConfig(transport="udp", host="192.168.1.50", udp_port=9000)
    assert a.transport == "udp" and a.host == "192.168.1.50" and a.udp_port == 9000


def test_rejects_bad_transport_and_port():
    with pytest.raises(ValidationError):
        ArduinoConfig(transport="ble")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ArduinoConfig(udp_port=70000)


def test_nested_backward_compatible():
    assert isinstance(AppConfig().arduino, ArduinoConfig)
    assert AppConfig(detection={"model": "nano"}).arduino.transport == "serial"
```

```python
# tests/aim/test_arduino_transport.py
"""Tests for the Arduino transport factory (CI-safe: no pyserial/socket reached)."""
from __future__ import annotations
import pytest
from ragnarok.config.schema import AppConfig
from ragnarok.aim import arduino


def test_module_imports_without_pyserial():
    import sys
    assert hasattr(arduino, "build_arduino_transport")
    assert "serial" not in sys.modules     # pyserial not imported by import alone


def test_build_serial_without_port_raises():
    cfg = AppConfig(arduino={"transport": "serial", "port": ""})
    with pytest.raises(RuntimeError, match="port"):
        arduino.build_arduino_transport(cfg)


def test_build_udp_without_host_raises():
    cfg = AppConfig(arduino={"transport": "udp", "host": "", "udp_port": 9000})
    with pytest.raises(RuntimeError, match="host"):
        arduino.build_arduino_transport(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_arduino_config.py tests/aim/test_arduino_transport.py -v`
Expected: FAIL — `ArduinoConfig` / `build_arduino_transport` not defined.

- [ ] **Step 3: Add ArduinoConfig and the transports/factory**

In `src/ragnarok/config/schema.py`, add (before `AppConfig`):

```python
class ArduinoConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    transport: Literal["serial", "udp"] = "serial"
    port: str = ""                                  # COM/tty for the serial transport
    baud: int = Field(default=115200, ge=1200)      # ignored on R4 native USB; for bridges
    host: str = ""                                  # IP for the UDP/WiFi transport
    udp_port: int = Field(default=0, ge=0, le=65535)
```

Add to `AppConfig` (after `training`):

```python
    arduino: ArduinoConfig = ArduinoConfig()
```

Append to `src/ragnarok/aim/arduino.py`:

```python
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


def build_arduino_transport(cfg):
    a = cfg.arduino
    if a.transport == "serial":
        if not a.port:
            raise RuntimeError("arduino.port must be set for the serial transport")
        return SerialTransport(a.port, a.baud)
    if not a.host or not a.udp_port:
        raise RuntimeError("arduino.host and arduino.udp_port must be set for the udp transport")
    return UdpTransport(a.host, a.udp_port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config tests/aim/test_arduino_transport.py -q`
Expected: PASS (guard tests raise before the lazy `import serial`; UDP guard raises before constructing).

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 7A).

```bash
git add src/ragnarok/config/schema.py src/ragnarok/aim/arduino.py tests/config/test_arduino_config.py tests/aim/test_arduino_transport.py
git commit -m "feat(aim,config): ArduinoConfig + serial/udp transport factory (lazy, box-only adapters)"
```

---

## Phase 7A completion checklist

- [ ] Wire protocol codec — frames + CRC8, all 4 CMDs, decode with validation (T1).
- [ ] `ArduinoDriver` — `MouseDriver` over an injected transport, sub-pixel accumulation + ±127 HID chunking + button mask (T2).
- [ ] `ArduinoConfig` + `build_arduino_transport` (serial/udp), guards before the lazy `pyserial` import; UDP via stdlib socket (T3).
- [ ] Full suite green; CI-safe (no serial/socket/MCU/pyserial/firmware in tests); SendInput stays default; Scope-Boundary deferrals (firmware `.ino`, BLE/ESP-NOW, real I/O, HIL timing loop, Interception) documented.

After merge: update memory (Phase 7A done — Arduino PC-side protocol + driver ready behind an injected transport). **Box-only smoke:** flash the matching firmware (mirrors this exact framing/CRC), set `arduino.transport`+`port`/`host`, `build_arduino_transport(cfg)`, wire `ArduinoDriver` into the controller's mouse slot (alternative to SendInput), verify moves on the device + HIL DIAG echo. Natural next: Phase 7B (firmware + BLE/ESP-NOW transports + passthrough), then Phase 8 (Cyberpunk GUI — consumes 5A diagnostics + apply_seeds, 5B calibration, and the input-driver selection).
