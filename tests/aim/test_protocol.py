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
