# Ragnarok Phase 6C — ONNX/TensorRT Export & Engine Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the training trilogy: build the CI-safe scaffolding to export a trained RF-DETR to a TensorRT engine and run it as a swappable detector backend (spec §5.2, §12 step 4), so the FP16-vs-INT8 comparison can be measured with the Phase-6A benchmark harness.

**Architecture:** A new `detection/export.py` constructs the engine-build command (stable `trtexec` flags) and runs it through an **injected runner** (CI asserts the command; the real run is box-only). Engine-path naming is a pure resolver. A new `RFDETRTensorRTDetector` implements the existing `Detector` ABC behind an injected **`Session` Protocol** — `detect()` maps the session's raw `(boxes, scores, classes)` arrays into our `Detections` and is fully unit-tested with a fake session; the real TensorRT engine load + inference is lazy/box-only. The detector `factory` selects torch vs TRT by `DetectionConfig.backend`. Nothing in this plan requires a GPU, `rfdetr`, `tensorrt`, or a trained checkpoint to test.

**Tech Stack:** Python 3.11+, stdlib (`subprocess`, `shutil`) for the real runner, numpy. `tensorrt` / `rfdetr` exporter / `trtexec` are **box-only** (lazy, never imported in tests). No new test dependencies.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is the deploy/export tooling for the training loop (spec §12).
- **CI-safe always:** no GPU / `tensorrt` / `rfdetr` / `trtexec` / trained checkpoint in unit tests. Export runs through an injected runner; the TRT detector runs through an injected session; the real engine load + the real ONNX export are lazy and box-only. Modules import without any of those packages.
- **FP16 is the production default** (spec §5.2/§18): Ampere (RTX 3090), no FP8. **INT8 is optional and box-only** — and real accuracy-preserving INT8 needs explicit Q/DQ via NVIDIA `modelopt` (the legacy `trtexec --int8` PTQ calibrators are deprecated/removed in TRT 11), so this plan only constructs the flag + documents the modelopt path as deferred; it does NOT claim a working INT8 calibration.
- **Single `player` class** (spec §5.2).
- **Reuse, don't reinvent:** the benchmark harness (`ragnarok.training.benchmark.run_benchmark`, Phase 6A) measures any `Detector` over a labeled set — FP16-vs-INT8 comparison is "run the harness with each engine," not new metric code. The `Detector` ABC + `Detections` types are consumed as-is.
- **Frozen pydantic config**, backward-compatible TOML round-trip (existing config without the new fields still loads).
- **TDD, frequent commits, exact file paths.** Match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, lazy heavy imports like `rfdetr_torch.py`, module docstrings, injected collaborators).

## Scope Boundary (explicit deferrals)

- **Real ONNX export** (RF-DETR `.pth` → `.onnx` via rfdetr's exporter) → box-only. The exporter API varies by rfdetr version, so this plan provides the *seam* (an injected exporter callable) and a documented box-only adapter, not a tested call. It produces the `.onnx` the TRT step consumes.
- **Real TensorRT engine build** (`trtexec` execution) and **real engine inference** (TRT runtime / CUDA) → box-only. CI tests the command construction + the detect() array→Detections mapping behind fakes.
- **INT8 explicit-Q/DQ calibration via `modelopt`** (spec §5.2) → deferred; FP16 is the default. This plan only emits the `--int8` flag + documents that proper INT8 needs the modelopt QAT/PTQ workflow.
- **2:4 sparsity** (spec experimental-only) → out of scope.
- **ONNX Runtime portable backend** (spec fallback) → out of scope (YAGNI; the deploy target is the TRT engine).
- **Hot-swapping engines into a live worker at runtime** → box-only follow-up; the factory selection + config are the seam.

---

## File Structure

**New files:**
- `src/ragnarok/detection/export.py` — `engine_path_for`, `build_trt_command`, `export_engine` (injected runner), `export_onnx` (injected exporter seam).
- `src/ragnarok/detection/rfdetr_trt.py` — `Session` Protocol + `RFDETRTensorRTDetector` (injected session) + the lazy real-session builder.
- `tests/detection/test_export.py`, `tests/detection/test_rfdetr_trt.py`

**Modified files:**
- `src/ragnarok/config/schema.py` — `DetectionConfig.backend` Literal gains `"rfdetr_trt"`; add `engine_path: str = ""`, `precision: Literal["fp16","int8"] = "fp16"`.
- `src/ragnarok/detection/factory.py` — select `rfdetr_torch` vs `rfdetr_trt` by `config.backend`.
- `tests/config/test_*` (extend for the new DetectionConfig fields).
- `tests/detection/test_base.py` or a factory test (factory selection).

---

## Task 1: DetectionConfig — TRT backend + engine fields

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Test: `tests/detection/test_detection_config.py` (new; if a DetectionConfig test already exists, extend it instead)

**Interfaces:**
- Produces: `DetectionConfig.backend: Literal["rfdetr_torch", "rfdetr_trt"] = "rfdetr_torch"`; new fields `engine_path: str = ""` (path to a built `.engine`), `precision: Literal["fp16", "int8"] = "fp16"`. `optimize_fp16` and the rest stay unchanged. Backward compatible (defaults keep `rfdetr_torch`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_detection_config.py
"""Tests for the Phase 6C DetectionConfig additions."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DetectionConfig, AppConfig


def test_backend_defaults_torch():
    assert DetectionConfig().backend == "rfdetr_torch"


def test_trt_backend_and_fields():
    d = DetectionConfig(backend="rfdetr_trt", engine_path="e.engine", precision="int8")
    assert d.backend == "rfdetr_trt"
    assert d.engine_path == "e.engine"
    assert d.precision == "int8"


def test_precision_default_fp16():
    assert DetectionConfig().precision == "fp16"


def test_bad_backend_rejected():
    with pytest.raises(ValidationError):
        DetectionConfig(backend="onnxruntime")  # type: ignore[arg-type]


def test_bad_precision_rejected():
    with pytest.raises(ValidationError):
        DetectionConfig(precision="fp8")  # type: ignore[arg-type]


def test_backward_compatible_in_appconfig():
    app = AppConfig(detection={"model": "nano"})
    assert app.detection.backend == "rfdetr_torch"
    assert app.detection.engine_path == "" and app.detection.precision == "fp16"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_detection_config.py -v`
Expected: FAIL — `DetectionConfig` rejects `backend="rfdetr_trt"` / has no `engine_path`.

- [ ] **Step 3: Extend DetectionConfig**

In `src/ragnarok/config/schema.py`, change `DetectionConfig`:

```python
class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["rfdetr_torch", "rfdetr_trt"] = "rfdetr_torch"
    model: Literal["nano", "small", "medium", "large"] = "small"  # Apache-2.0 variants only
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    optimize_fp16: bool = True   # fuse + FP16 the auto-built model (~order-of-magnitude on Ampere)
    engine_path: str = ""        # path to a built TensorRT .engine (rfdetr_trt backend)
    precision: Literal["fp16", "int8"] = "fp16"   # FP16 default; INT8 box-only (needs modelopt Q/DQ)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config tests/detection/test_detection_config.py -q`
Expected: PASS (incl. existing detector tests + TOML round-trip).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py tests/detection/test_detection_config.py
git commit -m "feat(config): DetectionConfig rfdetr_trt backend + engine_path/precision"
```

---

## Task 2: Export orchestration (engine path, trtexec command, runners)

**Files:**
- Create: `src/ragnarok/detection/export.py`
- Create: `tests/detection/test_export.py`

**Interfaces:**
- Consumes: nothing (stdlib).
- Produces:
  - `engine_path_for(engines_dir: str, model: str, precision: str) -> str` — deterministic naming: `"{engines_dir}/rfdetr-{model}-{precision}.engine"` using forward slashes (posix-style; the caller may normalize).
  - `build_trt_command(onnx_path: str, engine_path: str, *, precision: str = "fp16") -> list[str]` — the `trtexec` argv: `["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"]`, and for `precision == "int8"` ALSO append `"--int8"` (FP16 stays on as the mixed-precision fallback). Raises `ValueError` on an unknown precision.
  - `export_engine(onnx_path, engine_path, *, precision="fp16", runner=None) -> str` — builds the command and executes it via `runner(cmd) -> int` (returns the engine_path on success / raises `RuntimeError` on non-zero). `runner=None` → a real `subprocess`-based runner (box-only). CI injects a fake runner.
  - `export_onnx(model, onnx_path, *, exporter) -> str` — calls `exporter(model, onnx_path)` and returns `onnx_path`. The `exporter` is injected; the real one wraps rfdetr's `.export()` (box-only). This is the seam that produces the `.onnx` `build_trt_command` consumes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_export.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_export.py -v`
Expected: FAIL — `No module named 'ragnarok.detection.export'`.

- [ ] **Step 3: Implement export.py**

```python
# src/ragnarok/detection/export.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/detection/test_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/detection/export.py tests/detection/test_export.py
git commit -m "feat(detection): TRT export orchestration (path/command pure, runner/exporter injected)"
```

---

## Task 3: TensorRT engine detector + factory selection

**Files:**
- Create: `src/ragnarok/detection/rfdetr_trt.py`
- Modify: `src/ragnarok/detection/factory.py`
- Create: `tests/detection/test_rfdetr_trt.py`

**Interfaces:**
- Consumes: `Detector`/`Detections`/`Detection` (`ragnarok.detection.base`, `ragnarok.core.types`); `Frame`; `DetectionConfig` (T1).
- Produces:
  - `Session` (`typing.Protocol`): `infer(self, image, *, threshold: float) -> tuple[list, list, list]` — returns `(boxes_xyxy, scores, class_ids)` parallel sequences for one image.
  - `RFDETRTensorRTDetector(config, *, session=None)` implementing `Detector`: `detect(frame)` converts BGR→RGB (cv2), calls `session.infer(rgb, threshold=config.confidence)`, builds `Detections` directly from the three arrays. `session=None` → lazily build a real TRT session from `config.engine_path` (box-only). CI injects a fake session.
  - `create_detector(config)` (factory) selects `RFDETRTensorRTDetector` when `config.backend == "rfdetr_trt"`, else `RFDETRTorchDetector`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_rfdetr_trt.py
"""Tests for the TensorRT engine detector against a fake session (no GPU/engine)."""
from __future__ import annotations
import numpy as np
from ragnarok.config.schema import DetectionConfig
from ragnarok.core.types import Frame
from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector
from ragnarok.detection.factory import create_detector


class _FakeSession:
    def __init__(self):
        self.threshold = None
    def infer(self, image, *, threshold):
        self.threshold = threshold
        return ([(10.0, 10.0, 20.0, 30.0)], [0.95], [0])   # boxes, scores, classes


def _frame():
    return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))


def test_detect_maps_session_arrays_to_detections():
    sess = _FakeSession()
    det = RFDETRTensorRTDetector(DetectionConfig(backend="rfdetr_trt", confidence=0.6), session=sess)
    out = det.detect(_frame())
    assert len(out) == 1
    d = list(out)[0]
    assert d.xyxy == (10.0, 10.0, 20.0, 30.0) and d.confidence == 0.95 and d.class_id == 0
    assert sess.threshold == 0.6                  # confidence threaded into the session


def test_detect_empty_session_yields_no_detections():
    class _Empty:
        def infer(self, image, *, threshold):
            return ([], [], [])
    out = RFDETRTensorRTDetector(DetectionConfig(backend="rfdetr_trt"), session=_Empty()).detect(_frame())
    assert len(out) == 0


def test_factory_routes_trt_backend():
    # The factory must select the TRT class for backend=rfdetr_trt. With no
    # injected session it calls _build_trt_session, which raises a DISTINCT
    # NotImplementedError (box-only) — proving the TRT route specifically (a wrong
    # torch route would raise ModuleNotFoundError from the lazy `import rfdetr`).
    import pytest
    cfg = DetectionConfig(backend="rfdetr_trt", engine_path="missing.engine")
    with pytest.raises(NotImplementedError):
        create_detector(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_rfdetr_trt.py -v`
Expected: FAIL — `No module named 'ragnarok.detection.rfdetr_trt'`.

- [ ] **Step 3: Implement rfdetr_trt.py and the factory selection**

```python
# src/ragnarok/detection/rfdetr_trt.py
"""TensorRT engine detector (spec §5.2, §12.4).

detect() maps an injected Session's raw (boxes, scores, classes) into Detections,
fully unit-testable with a fake session. The real TensorRT engine load + inference
is lazy and box-only (needs the .engine + tensorrt + CUDA).
"""
from __future__ import annotations

from typing import Protocol

import cv2

from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector


class Session(Protocol):
    def infer(self, image, *, threshold: float) -> tuple[list, list, list]: ...


class RFDETRTensorRTDetector(Detector):
    def __init__(self, config: DetectionConfig, *, session: Session | None = None) -> None:
        self._config = config
        if session is None:
            session = _build_trt_session(config.engine_path)  # lazy/box-only
        self._session = session

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = self._session.infer(rgb, threshold=self._config.confidence)
        items = tuple(
            Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)),
                      confidence=float(s), class_id=int(c))
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes)
        )
        return Detections(items=items)


def _build_trt_session(engine_path: str) -> Session:  # pragma: no cover — box-only
    """Load a TensorRT engine into an inference Session. Box-only (tensorrt + CUDA)."""
    raise NotImplementedError(
        "Real TensorRT session loading is box-only; inject a Session in tests / "
        "implement the tensorrt runtime adapter on the deployment box "
        f"(engine_path={engine_path!r})."
    )
```

In `src/ragnarok/detection/factory.py`:

```python
from __future__ import annotations
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector


def create_detector(config: DetectionConfig) -> Detector:
    if config.backend == "rfdetr_trt":
        from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector
        return RFDETRTensorRTDetector(config)
    from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector
    return RFDETRTorchDetector(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/detection -q`
Expected: PASS (TRT detect tests + factory routing; existing torch tests unaffected).

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 6C).

```bash
git add src/ragnarok/detection/rfdetr_trt.py src/ragnarok/detection/factory.py tests/detection/test_rfdetr_trt.py
git commit -m "feat(detection): TensorRT engine detector (injected session) + factory selection"
```

---

## Phase 6C completion checklist

- [ ] `DetectionConfig` TRT backend + `engine_path`/`precision` (T1).
- [ ] Export orchestration: pure `engine_path_for`/`build_trt_command` + injected-runner `export_engine` + injected-exporter `export_onnx` (T2).
- [ ] `RFDETRTensorRTDetector` (injected `Session`) + factory selection; `detect()` maps arrays→Detections (T3).
- [ ] Full suite green; CI-safe (no GPU/tensorrt/rfdetr/trtexec/checkpoint in tests); FP16 default, INT8 flag-only with the modelopt path documented as deferred.
- [ ] Scope-Boundary deferrals (real ONNX export, real trtexec build, real TRT runtime session, INT8 modelopt calibration, 2:4 sparsity, ONNX Runtime backend, live hot-swap) documented.

After merge: update memory (Phase 6C done — training trilogy complete: collect/measure (6A) → Roboflow (6B) → export/deploy (6C)). **Box-only smoke to use it end-to-end:** train RF-DETR on the labeled Roboflow dataset → `export_onnx` (real rfdetr exporter) → `export_engine` (real trtexec, FP16) → set `detection.backend="rfdetr_trt"` + `engine_path` → implement `_build_trt_session` (tensorrt runtime adapter) → `run_benchmark` (6A) FP16 vs the torch baseline on the user's frames. Natural next: 5C (dynamic-ROI), Phase 7 (Arduino backends), or Phase 8 (Cyberpunk GUI — consumes 5A diagnostics + apply_seeds, 5B calibration).
