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
