"""Tests for the Roboflow transport factory (CI-safe: no roboflow import reached)."""
from __future__ import annotations
import pytest
from ragnarok.config.schema import AppConfig
from ragnarok.training import roboflow_client as rc


def test_module_imports_without_roboflow_installed():
    # The real SDK import is lazy (inside RoboflowSdkTransport), so importing the
    # module must succeed even though `roboflow` is not a test dependency.
    assert hasattr(rc, "RoboflowSdkTransport")
    assert hasattr(rc, "build_roboflow_transport")


def test_build_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("RAGNAROK_ROBOFLOW_API_KEY", raising=False)
    cfg = AppConfig(training={"roboflow_workspace": "ws", "roboflow_project": "proj"})
    with pytest.raises(RuntimeError, match="RAGNAROK_ROBOFLOW_API_KEY"):
        rc.build_roboflow_transport(cfg)


def test_build_raises_without_workspace_or_project(monkeypatch):
    monkeypatch.setenv("RAGNAROK_ROBOFLOW_API_KEY", "k")
    cfg = AppConfig()                                # workspace/project default ""
    with pytest.raises(RuntimeError, match="roboflow_workspace"):
        rc.build_roboflow_transport(cfg)
