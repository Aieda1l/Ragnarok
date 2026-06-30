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
