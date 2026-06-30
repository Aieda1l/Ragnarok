"""Tests for the export orchestration (injected runner/exporter — no GPU/trtexec)."""
from __future__ import annotations
import pytest
from ragnarok.detection.export import engine_path_for, build_trt_command, export_engine, export_onnx


def test_engine_path_naming():
    assert engine_path_for("engines", "small", "fp16") == "engines/rfdetr-small-fp16.engine"


def test_trt_command_fp16():
    cmd = build_trt_command("m.onnx", "m.engine", precision="fp16")
    assert cmd == ["trtexec", "--onnx=m.onnx", "--saveEngine=m.engine", "--fp16"]


def test_trt_command_int8_keeps_fp16_fallback():
    cmd = build_trt_command("m.onnx", "m.engine", precision="int8")
    assert "--int8" in cmd and "--fp16" in cmd


def test_trt_command_bad_precision():
    with pytest.raises(ValueError):
        build_trt_command("m.onnx", "m.engine", precision="fp8")


def test_export_engine_runs_command_via_runner():
    calls = []
    def runner(cmd):
        calls.append(cmd)
        return 0
    out = export_engine("m.onnx", "m.engine", precision="fp16", runner=runner)
    assert out == "m.engine"
    assert calls == [["trtexec", "--onnx=m.onnx", "--saveEngine=m.engine", "--fp16"]]


def test_export_engine_raises_on_nonzero():
    with pytest.raises(RuntimeError):
        export_engine("m.onnx", "m.engine", runner=lambda cmd: 1)


def test_export_onnx_invokes_exporter():
    seen = []
    out = export_onnx("MODEL", "m.onnx", exporter=lambda m, p: seen.append((m, p)))
    assert out == "m.onnx"
    assert seen == [("MODEL", "m.onnx")]
