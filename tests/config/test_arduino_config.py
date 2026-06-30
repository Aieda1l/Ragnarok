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
