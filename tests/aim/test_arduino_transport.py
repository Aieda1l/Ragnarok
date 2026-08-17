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


def test_build_hid_transport_routes():
    cfg = AppConfig(arduino={"transport": "hid", "vid": 0x2341, "hid_pid": 0x0069})
    t = arduino.build_arduino_transport(cfg)
    assert isinstance(t, arduino.HidTransport)


def test_build_hid_without_ids_raises():
    cfg = AppConfig(arduino={"transport": "hid", "vid": 0, "hid_pid": 0})
    with pytest.raises(RuntimeError, match="vid"):
        arduino.build_arduino_transport(cfg)


def test_hid_transport_writes_padded_report():
    class _FakeDev:
        def __init__(self): self.reports = []
        def write(self, data): self.reports.append(bytes(data)); return len(data)
        def close(self): pass

    dev = _FakeDev()
    t = arduino.HidTransport(vid=0x2341, pid=0x0069)
    t._dev = dev                     # inject the opened device (bypass real hidapi)
    frame = b"\xAA\x01\x06\x00" + b"\x00" * 6 + b"\x11"
    t.write(frame)
    assert len(dev.reports) == 1
    report = dev.reports[0]
    assert report[0] == 0x00                       # HID report id prefix
    assert report[1:1 + len(frame)] == frame       # frame carried verbatim
    assert len(report) == 1 + 64                   # padded to the fixed report length
