"""RF-DETR -> ONNX -> TensorRT engine export orchestration (spec §5.2, §12.4).

CI-safe: command construction + path resolution are pure; the actual engine
build runs through an injected `runner` and the ONNX export through an injected
`exporter`, so unit tests never invoke trtexec / rfdetr / a GPU. The real
runner/exporter are box-only. FP16 is the default; INT8 only emits the flag —
accuracy-preserving INT8 needs the NVIDIA modelopt Q/DQ workflow (deferred).
"""
from __future__ import annotations

_VALID_PRECISION = ("fp16", "int8")


def engine_path_for(engines_dir: str, model: str, precision: str) -> str:
    return f"{engines_dir}/rfdetr-{model}-{precision}.engine"


def build_trt_command(onnx_path: str, engine_path: str, *, precision: str = "fp16") -> list[str]:
    if precision not in _VALID_PRECISION:
        raise ValueError(f"unknown precision {precision!r}; choose from {_VALID_PRECISION}")
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"]
    if precision == "int8":
        cmd.append("--int8")   # mixed INT8+FP16; real calibration is modelopt Q/DQ (box-only)
    return cmd


def _subprocess_runner(cmd) -> int:  # pragma: no cover — box-only (real trtexec)
    import subprocess
    return subprocess.run(cmd, check=False).returncode


def export_engine(onnx_path: str, engine_path: str, *, precision: str = "fp16",
                  runner=None) -> str:
    run = runner if runner is not None else _subprocess_runner
    cmd = build_trt_command(onnx_path, engine_path, precision=precision)
    code = run(cmd)
    if code != 0:
        raise RuntimeError(f"engine build failed (exit {code}): {' '.join(cmd)}")
    return engine_path


def export_onnx(model, onnx_path: str, *, exporter) -> str:
    """Export a torch RF-DETR model to ONNX via an injected exporter callable.

    The real exporter wraps rfdetr's own export (box-only; API varies by version).
    """
    exporter(model, onnx_path)
    return onnx_path
