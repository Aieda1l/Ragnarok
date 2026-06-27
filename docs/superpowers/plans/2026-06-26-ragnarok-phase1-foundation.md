# Ragnarok Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Ragnarok skeleton end-to-end: capture a screen ROI → run RF-DETR detection → profile per-stage latency → publish a telemetry snapshot to a minimal PySide6 GUI, with all heavy work off the GUI thread.

**Architecture:** One process, two threads. A worker thread owns capture + detection + profiling and publishes an immutable telemetry snapshot via a single atomic reference; the Qt GUI thread polls that snapshot on a 60 Hz `QTimer` and renders a preview + latency readout. Everything is behind narrow interfaces (`Capturer`, `Detector`) with dependency injection so the hot path can be unit-tested with fakes (no GPU/display/weights needed for tests).

**Tech Stack:** Python 3.11+, PySide6, `bettercam` (+ `mss` fallback), `rfdetr` + `supervision`, NumPy, pydantic v2, tomlkit, pytest + pytest-qt.

**Scope of this phase (spec §17.1):** project skeleton, config system, capture, RF-DETR detection (Torch/ONNX baseline — TensorRT engine + dynamic-ROI come in Phase 5), `perf_counter_ns` timestamping, per-stage latency profiler, telemetry snapshot plumbing, and the GUI shell. **Out of scope:** tracking, friend/foe, aim, output drivers, recoil, trigger, training, overlay (later phases).

This plan implements only the subset of the spec needed for a working foundation. References to the spec mean `docs/superpowers/specs/2026-06-26-ragnarok-design.md`.

## Global Constraints

- **Python 3.11+** — required (`time.perf_counter_ns` = QPC; `tomllib` in stdlib). One line each below is project-wide; every task implicitly includes these.
- **Platform:** Windows 11, single-PC. (Tests must still run headless/CI-safe via fakes.)
- **GPU:** RTX 3090. Phase 1 detection runs the **Torch FP16** RF-DETR path; the TensorRT FP16 engine + dynamic-ROI are Phase 5. **FP16 is the default precision.**
- **Threading:** ALL CUDA/torch/capture state lives in the **worker thread only**. The GUI thread never touches CUDA.
- **Worker→GUI telemetry:** publish-latest **immutable** snapshot via a single atomic reference assignment; GUI **polls** on a ≤60 Hz `QTimer`. **No Qt signals on the hot path.**
- **Config:** pydantic-validated, immutable snapshots swapped atomically; persisted as **TOML** under `%APPDATA%/Ragnarok`.
- **Timestamps:** `time.perf_counter_ns()` integer nanoseconds, stamped at capture, propagated inside the `Frame`. Convert to ms only for display.
- **Detector model license:** RF-DETR **Apache-2.0** variants only (Nano/Small/Medium/Large) — never XL/2XL.
- **Process:** TDD (test first), DRY, YAGNI, frequent commits (one per task).
- **Package import root:** `ragnarok` (installed editable via `pip install -e .`). All tests import from `ragnarok.*`.

---

## File Structure

Created in this phase:

```
pyproject.toml                              # project metadata, deps, pytest config
src/ragnarok/__init__.py
src/ragnarok/core/__init__.py
src/ragnarok/core/clock.py                  # now_ns(), ns_to_ms()
src/ragnarok/core/types.py                  # Frame, Detection, Detections (frozen dataclasses)
src/ragnarok/config/__init__.py
src/ragnarok/config/schema.py               # pydantic: CaptureConfig, DetectionConfig, AppConfig
src/ragnarok/config/store.py                # TOML load/save + ConfigHandle (atomic snapshot swap)
src/ragnarok/latency/__init__.py
src/ragnarok/latency/profiler.py            # StageProfiler (ring buffers, p50/p99)
src/ragnarok/telemetry/__init__.py
src/ragnarok/telemetry/snapshot.py          # TelemetrySnapshot + SnapshotPublisher (publish-latest)
src/ragnarok/capture/__init__.py
src/ragnarok/capture/base.py                # Capturer ABC + RegionSpec
src/ragnarok/capture/mss_capturer.py        # portable fallback
src/ragnarok/capture/bettercam_capturer.py  # DXGI default (injectable bettercam module)
src/ragnarok/capture/factory.py             # create_capturer(config)
src/ragnarok/detection/__init__.py
src/ragnarok/detection/base.py              # Detector ABC + to_detections() mapping helper
src/ragnarok/detection/rfdetr_torch.py      # RF-DETR (Torch) detector (injectable model)
src/ragnarok/detection/factory.py           # create_detector(config)
src/ragnarok/worker/__init__.py
src/ragnarok/worker/loop.py                 # WorkerLoop: capture->detect->profile->publish
src/ragnarok/gui/__init__.py
src/ragnarok/gui/worker_thread.py           # QThread runner around WorkerLoop
src/ragnarok/gui/main_window.py             # preview + FPS/latency, QTimer poll
src/ragnarok/app.py                         # entry point wiring
tests/ (mirrors src tree)
```

**Responsibilities:** `core` = shared value types + clock. `config` = typed settings + persistence + atomic snapshots. `capture`/`detection` = swappable interfaces with one impl each (+ fakes in tests). `latency` = measurement. `telemetry` = the worker→GUI handoff. `worker` = the hot loop (pure, GUI-agnostic). `gui` = Qt shell + thread plumbing. `app` = composition root.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/ragnarok/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `ragnarok` package (`ragnarok.__version__: str`); `pytest` runnable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import ragnarok

def test_package_has_version():
    assert isinstance(ragnarok.__version__, str)
    assert ragnarok.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ragnarok"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pydantic>=2.6",
    "tomlkit>=0.12",
    "PySide6>=6.6",
    "bettercam>=1.0; sys_platform == 'win32'",
    "mss>=9.0",
    "supervision>=0.20",
    # rfdetr + torch are installed separately per environment (CUDA build); see README.
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

```python
# src/ragnarok/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Install editable and run the test**

Run: `pip install -e ".[dev]"` then `pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ragnarok/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold ragnarok package and pytest"
```

---

### Task 2: Monotonic clock

**Files:**
- Create: `src/ragnarok/core/__init__.py`, `src/ragnarok/core/clock.py`, `tests/core/__init__.py`, `tests/core/test_clock.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `now_ns() -> int` (QPC-backed monotonic ns); `ns_to_ms(ns: int) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_clock.py
from ragnarok.core.clock import now_ns, ns_to_ms

def test_now_ns_is_monotonic_int():
    a = now_ns()
    b = now_ns()
    assert isinstance(a, int) and isinstance(b, int)
    assert b >= a

def test_ns_to_ms():
    assert ns_to_ms(1_500_000) == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_clock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.core'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/core/__init__.py
```

```python
# src/ragnarok/core/clock.py
"""Monotonic, QPC-backed timing. perf_counter_ns is monotonic and NTP-immune;
only differences are meaningful (undefined epoch)."""
import time

def now_ns() -> int:
    return time.perf_counter_ns()

def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0
```

```python
# tests/core/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_clock.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/core/ tests/core/
git commit -m "feat(core): add monotonic clock helpers"
```

---

### Task 3: Core value types

**Files:**
- Create: `src/ragnarok/core/types.py`, `tests/core/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Frame(image: np.ndarray, t_capture_ns: int, region: tuple[int,int,int,int])` (frozen).
  - `Detection(xyxy: tuple[float,float,float,float], confidence: float, class_id: int)` (frozen); property `center -> tuple[float,float]`.
  - `Detections(items: tuple[Detection, ...])` (frozen); `__len__`, `__iter__`, classmethod `empty() -> Detections`.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_types.py
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections

def test_frame_holds_image_and_timestamp():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    f = Frame(image=img, t_capture_ns=123, region=(0, 0, 4, 4))
    assert f.t_capture_ns == 123
    assert f.image.shape == (4, 4, 3)

def test_detection_center():
    d = Detection(xyxy=(10.0, 20.0, 30.0, 60.0), confidence=0.9, class_id=0)
    assert d.center == (20.0, 40.0)

def test_detections_container():
    empty = Detections.empty()
    assert len(empty) == 0
    one = Detections(items=(Detection((0, 0, 2, 2), 0.5, 0),))
    assert len(one) == 1
    assert list(one)[0].confidence == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_types.py -v`
Expected: FAIL — `ImportError: cannot import name 'Frame'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/core/types.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Frame:
    image: np.ndarray            # HxWx3 uint8, BGR
    t_capture_ns: int            # now_ns() at grab
    region: tuple[int, int, int, int]  # (left, top, right, bottom) absolute screen coords

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]  # in ROI pixel coords
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

@dataclass(frozen=True)
class Detections:
    items: tuple[Detection, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(items=())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/core/types.py tests/core/test_types.py
git commit -m "feat(core): add Frame/Detection/Detections value types"
```

---

### Task 4: Config schema

**Files:**
- Create: `src/ragnarok/config/__init__.py`, `src/ragnarok/config/schema.py`, `tests/config/__init__.py`, `tests/config/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces (pydantic v2 `BaseModel`, all frozen via `model_config = ConfigDict(frozen=True)`):
  - `CaptureConfig(backend: Literal["bettercam","mss"]="bettercam", roi_size: int=384, target_fps: int=144, monitor_index: int=0)`.
  - `DetectionConfig(backend: Literal["rfdetr_torch"]="rfdetr_torch", model: Literal["nano","small","medium","large"]="small", confidence: float=0.5)`.
  - `AppConfig(capture: CaptureConfig=CaptureConfig(), detection: DetectionConfig=DetectionConfig())`.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_schema.py
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig, CaptureConfig, DetectionConfig

def test_defaults():
    cfg = AppConfig()
    assert cfg.capture.roi_size == 384
    assert cfg.detection.model == "small"
    assert cfg.detection.confidence == 0.5

def test_is_frozen():
    cfg = CaptureConfig()
    with pytest.raises(ValidationError):
        cfg.roi_size = 512  # frozen -> ValidationError

def test_rejects_bad_model():
    with pytest.raises(ValidationError):
        DetectionConfig(model="xl")  # not an Apache-2.0 variant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/config/__init__.py
```

```python
# src/ragnarok/config/schema.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["bettercam", "mss"] = "bettercam"
    roi_size: int = Field(default=384, ge=64, le=1280)
    target_fps: int = Field(default=144, ge=1, le=1000)
    monitor_index: int = 0

class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["rfdetr_torch"] = "rfdetr_torch"
    model: Literal["nano", "small", "medium", "large"] = "small"  # Apache-2.0 variants only
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    capture: CaptureConfig = CaptureConfig()
    detection: DetectionConfig = DetectionConfig()
```

```python
# tests/config/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/ tests/config/
git commit -m "feat(config): add pydantic settings schema"
```

---

### Task 5: Config store (TOML + atomic snapshot)

**Files:**
- Create: `src/ragnarok/config/store.py`, `tests/config/test_store.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 4).
- Produces:
  - `load_config(path: Path) -> AppConfig` (returns defaults + writes file if missing).
  - `save_config(cfg: AppConfig, path: Path) -> None`.
  - `ConfigHandle` — holds the live snapshot: `.current -> AppConfig` (atomic read), `.swap(cfg: AppConfig) -> None` (atomic write). The swap is a single attribute rebind (GIL-atomic; no lock).

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_store.py
from ragnarok.config.schema import AppConfig, CaptureConfig
from ragnarok.config.store import load_config, save_config, ConfigHandle

def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "config.toml"
    cfg = AppConfig(capture=CaptureConfig(roi_size=512, target_fps=240))
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.capture.roi_size == 512
    assert loaded.capture.target_fps == 240

def test_load_missing_writes_defaults(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config(p)
    assert p.exists()
    assert cfg.capture.roi_size == 384

def test_handle_atomic_swap():
    h = ConfigHandle(AppConfig())
    assert h.current.capture.roi_size == 384
    h.swap(AppConfig(capture=CaptureConfig(roi_size=256)))
    assert h.current.capture.roi_size == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/config/store.py
from __future__ import annotations
from pathlib import Path
import tomlkit
from ragnarok.config.schema import AppConfig

def load_config(path: Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        cfg = AppConfig()
        save_config(cfg, path)
        return cfg
    data = tomlkit.parse(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(dict(data))

def save_config(cfg: AppConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(cfg.model_dump()), encoding="utf-8")

class ConfigHandle:
    """Live config snapshot. Single-writer/single-reader: `swap` rebinds one
    attribute (GIL-atomic), readers always see a whole AppConfig, never torn."""
    def __init__(self, initial: AppConfig) -> None:
        self._current = initial

    @property
    def current(self) -> AppConfig:
        return self._current

    def swap(self, cfg: AppConfig) -> None:
        self._current = cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/store.py tests/config/test_store.py
git commit -m "feat(config): add TOML store and atomic config handle"
```

---

### Task 6: Latency profiler

**Files:**
- Create: `src/ragnarok/latency/__init__.py`, `src/ragnarok/latency/profiler.py`, `tests/latency/__init__.py`, `tests/latency/test_profiler.py`

**Interfaces:**
- Consumes: nothing (uses raw ns ints).
- Produces:
  - `StageProfiler(window: int = 240)` — `record(stage: str, dt_ns: int) -> None`; `percentiles(stage: str) -> tuple[float, float]` returns `(p50_ms, p99_ms)`; `stages() -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/latency/test_profiler.py
from ragnarok.latency.profiler import StageProfiler

def test_percentiles_basic():
    p = StageProfiler(window=100)
    for i in range(100):
        p.record("infer", (i + 1) * 1_000_000)  # 1ms..100ms
    p50, p99 = p.percentiles("infer")
    assert 49.0 <= p50 <= 52.0
    assert p99 >= 98.0

def test_unknown_stage_returns_zeros():
    p = StageProfiler()
    assert p.percentiles("nope") == (0.0, 0.0)

def test_window_evicts_old():
    p = StageProfiler(window=3)
    for v in [10, 20, 30, 40]:  # ms
        p.record("s", v * 1_000_000)
    p50, _ = p.percentiles("s")
    assert p50 == 30.0  # only [20,30,40] retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/latency/test_profiler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.latency'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/latency/__init__.py
```

```python
# src/ragnarok/latency/profiler.py
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np

class StageProfiler:
    def __init__(self, window: int = 240) -> None:
        self._window = window
        self._samples: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, stage: str, dt_ns: int) -> None:
        self._samples[stage].append(int(dt_ns))

    def percentiles(self, stage: str) -> tuple[float, float]:
        buf = self._samples.get(stage)
        if not buf:
            return (0.0, 0.0)
        arr = np.fromiter(buf, dtype=np.int64)
        p50, p99 = np.percentile(arr, [50, 99])
        return (float(p50) / 1e6, float(p99) / 1e6)

    def stages(self) -> list[str]:
        return list(self._samples.keys())
```

```python
# tests/latency/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/latency/test_profiler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/latency/ tests/latency/
git commit -m "feat(latency): add per-stage p50/p99 profiler"
```

---

### Task 7: Telemetry snapshot + publisher

**Files:**
- Create: `src/ragnarok/telemetry/__init__.py`, `src/ragnarok/telemetry/snapshot.py`, `tests/telemetry/__init__.py`, `tests/telemetry/test_snapshot.py`

**Interfaces:**
- Consumes: `Detections` (Task 3) — count only here.
- Produces:
  - `TelemetrySnapshot(fps: float, loop_ms_p50: float, loop_ms_p99: float, detection_count: int, preview: np.ndarray | None, seq: int)` (frozen).
  - `SnapshotPublisher` — `publish(snap: TelemetrySnapshot) -> None` (atomic rebind); `latest() -> TelemetrySnapshot | None` (atomic read).

- [ ] **Step 1: Write the failing test**

```python
# tests/telemetry/test_snapshot.py
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher

def test_publisher_starts_empty():
    pub = SnapshotPublisher()
    assert pub.latest() is None

def test_publish_then_latest_returns_newest():
    pub = SnapshotPublisher()
    s1 = TelemetrySnapshot(fps=100.0, loop_ms_p50=5.0, loop_ms_p99=8.0,
                           detection_count=1, preview=None, seq=1)
    s2 = TelemetrySnapshot(fps=120.0, loop_ms_p50=4.0, loop_ms_p99=7.0,
                           detection_count=2, preview=None, seq=2)
    pub.publish(s1)
    pub.publish(s2)
    assert pub.latest().seq == 2
    assert pub.latest().fps == 120.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telemetry/test_snapshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.telemetry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/telemetry/__init__.py
```

```python
# src/ragnarok/telemetry/snapshot.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TelemetrySnapshot:
    fps: float
    loop_ms_p50: float
    loop_ms_p99: float
    detection_count: int
    preview: np.ndarray | None   # small BGR image for the GUI, or None
    seq: int

class SnapshotPublisher:
    """Single-writer (worker) / single-reader (GUI). publish() rebinds one
    attribute -> GIL-atomic; the reader gets a whole snapshot or None, never torn."""
    def __init__(self) -> None:
        self._latest: TelemetrySnapshot | None = None

    def publish(self, snap: TelemetrySnapshot) -> None:
        self._latest = snap

    def latest(self) -> TelemetrySnapshot | None:
        return self._latest
```

```python
# tests/telemetry/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telemetry/test_snapshot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/telemetry/ tests/telemetry/
git commit -m "feat(telemetry): add snapshot type and publish-latest publisher"
```

---

### Task 8: Capturer interface + mss fallback + factory

**Files:**
- Create: `src/ragnarok/capture/__init__.py`, `src/ragnarok/capture/base.py`, `src/ragnarok/capture/mss_capturer.py`, `src/ragnarok/capture/factory.py`, `tests/capture/__init__.py`, `tests/capture/test_base.py`, `tests/capture/test_factory.py`

**Interfaces:**
- Consumes: `Frame` (Task 3), `CaptureConfig` (Task 4), `now_ns` (Task 2).
- Produces:
  - `centered_region(roi_size, screen_w, screen_h) -> tuple[int,int,int,int]` (left, top, right, bottom).
  - `Capturer` ABC: `start() -> None`, `grab() -> Frame | None`, `stop() -> None`.
  - `MssCapturer(config, screen_size: tuple[int,int])`.
  - `create_capturer(config: CaptureConfig) -> Capturer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/test_base.py
import numpy as np
from ragnarok.capture.base import centered_region, Capturer
from ragnarok.core.types import Frame

def test_centered_region_math():
    # 384 ROI centered on a 1920x1080 screen
    assert centered_region(384, 1920, 1080) == (768, 348, 1152, 732)

def test_capturer_is_abstract():
    assert hasattr(Capturer, "grab")

class _FakeCapturer(Capturer):
    def start(self): self.started = True
    def grab(self):
        return Frame(image=np.zeros((4, 4, 3), np.uint8), t_capture_ns=1, region=(0, 0, 4, 4))
    def stop(self): self.started = False

def test_fake_capturer_grab_returns_frame():
    c = _FakeCapturer()
    c.start()
    f = c.grab()
    assert isinstance(f, Frame)
    c.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/capture/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.capture'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/capture/__init__.py
```

```python
# src/ragnarok/capture/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from ragnarok.core.types import Frame

def centered_region(roi_size: int, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
    half = roi_size // 2
    cx, cy = screen_w // 2, screen_h // 2
    return (cx - half, cy - half, cx + half, cy + half)

class Capturer(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def grab(self) -> Frame | None: ...
    @abstractmethod
    def stop(self) -> None: ...
```

```python
# src/ragnarok/capture/mss_capturer.py
from __future__ import annotations
import numpy as np
import mss
from ragnarok.core.clock import now_ns
from ragnarok.core.types import Frame
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer, centered_region

class MssCapturer(Capturer):
    def __init__(self, config: CaptureConfig, screen_size: tuple[int, int]) -> None:
        self._region = centered_region(config.roi_size, *screen_size)
        self._sct: mss.mss | None = None

    def start(self) -> None:
        self._sct = mss.mss()

    def grab(self) -> Frame | None:
        if self._sct is None:
            return None
        left, top, right, bottom = self._region
        bbox = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        raw = self._sct.grab(bbox)
        img = np.asarray(raw)[:, :, :3]  # BGRA -> BGR
        return Frame(image=np.ascontiguousarray(img), t_capture_ns=now_ns(), region=self._region)

    def stop(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
```

```python
# src/ragnarok/capture/factory.py
from __future__ import annotations
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer

def _screen_size() -> tuple[int, int]:
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance()
    if app is not None:
        geo = QGuiApplication.primaryScreen().geometry()
        return (geo.width(), geo.height())
    return (1920, 1080)  # safe default before a QApplication exists

def create_capturer(config: CaptureConfig) -> Capturer:
    size = _screen_size()
    if config.backend == "bettercam":
        from ragnarok.capture.bettercam_capturer import BetterCamCapturer
        return BetterCamCapturer(config, size)
    from ragnarok.capture.mss_capturer import MssCapturer
    return MssCapturer(config, size)
```

```python
# tests/capture/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/capture/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Write + run the factory test (mss path, no Windows/DXGI needed)**

```python
# tests/capture/test_factory.py
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.factory import create_capturer
from ragnarok.capture.mss_capturer import MssCapturer

def test_factory_returns_mss_for_mss_backend():
    cap = create_capturer(CaptureConfig(backend="mss"))
    assert isinstance(cap, MssCapturer)
```

Run: `pytest tests/capture/test_factory.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/capture/ tests/capture/
git commit -m "feat(capture): add Capturer interface, mss fallback, factory"
```

---

### Task 9: BetterCam capturer (DXGI default)

**Files:**
- Create: `src/ragnarok/capture/bettercam_capturer.py`, `tests/capture/test_bettercam.py`

**Interfaces:**
- Consumes: `Capturer`, `centered_region`, `Frame`, `now_ns`, `CaptureConfig`.
- Produces: `BetterCamCapturer(config, screen_size, *, bettercam_module=None)` — the `bettercam` module is injectable so the region/format logic is unit-tested with a fake (no GPU/display).

- [ ] **Step 1: Write the failing test**

```python
# tests/capture/test_bettercam.py
import numpy as np
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.bettercam_capturer import BetterCamCapturer
from ragnarok.core.types import Frame

class _FakeCam:
    def __init__(self): self.started_region = None; self.stopped = False
    def start(self, region, target_fps, video_mode=False):
        self.started_region = region; self.target_fps = target_fps
    def get_latest_frame(self):
        return np.zeros((384, 384, 3), dtype=np.uint8)  # BGR
    def stop(self): self.stopped = True

class _FakeModule:
    def __init__(self): self.cam = _FakeCam()
    def create(self, output_idx=0, output_color="BGR"): return self.cam

def test_start_uses_centered_region_and_fps():
    mod = _FakeModule()
    cap = BetterCamCapturer(CaptureConfig(roi_size=384, target_fps=144),
                            screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    assert mod.cam.started_region == (768, 348, 1152, 732)
    assert mod.cam.target_fps == 144

def test_grab_returns_frame_with_timestamp():
    mod = _FakeModule()
    cap = BetterCamCapturer(CaptureConfig(), screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    f = cap.grab()
    assert isinstance(f, Frame)
    assert f.t_capture_ns > 0
    assert f.image.shape == (384, 384, 3)

def test_grab_returns_none_when_no_new_frame():
    mod = _FakeModule()
    mod.cam.get_latest_frame = lambda: None
    cap = BetterCamCapturer(CaptureConfig(), screen_size=(1920, 1080), bettercam_module=mod)
    cap.start()
    assert cap.grab() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/capture/test_bettercam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.capture.bettercam_capturer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/capture/bettercam_capturer.py
from __future__ import annotations
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.core.types import Frame
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer, centered_region

class BetterCamCapturer(Capturer):
    def __init__(self, config: CaptureConfig, screen_size: tuple[int, int],
                 *, bettercam_module=None) -> None:
        self._config = config
        self._region = centered_region(config.roi_size, *screen_size)
        if bettercam_module is None:
            import bettercam  # imported lazily so tests don't need it
            bettercam_module = bettercam
        self._mod = bettercam_module
        self._cam = None

    def start(self) -> None:
        self._cam = self._mod.create(output_idx=self._config.monitor_index, output_color="BGR")
        self._cam.start(region=self._region, target_fps=self._config.target_fps, video_mode=False)

    def grab(self) -> Frame | None:
        if self._cam is None:
            return None
        img = self._cam.get_latest_frame()
        if img is None:
            return None
        return Frame(image=np.ascontiguousarray(img), t_capture_ns=now_ns(), region=self._region)

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/capture/test_bettercam.py -v`
Expected: PASS

- [ ] **Step 5: Manual smoke (real DXGI; not in CI)**

On the Windows box: `python -c "from ragnarok.config.schema import CaptureConfig; from ragnarok.capture.bettercam_capturer import BetterCamCapturer; c=BetterCamCapturer(CaptureConfig(),(1920,1080)); c.start(); import time; time.sleep(0.1); f=c.grab(); print(None if f is None else f.image.shape); c.stop()"`
Expected: prints `(384, 384, 3)` (or `None` on the first poll — re-run).

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/capture/bettercam_capturer.py tests/capture/test_bettercam.py
git commit -m "feat(capture): add BetterCam DXGI capturer (injectable module)"
```

---

### Task 10: Detector interface + RF-DETR (Torch) + factory

**Files:**
- Create: `src/ragnarok/detection/__init__.py`, `src/ragnarok/detection/base.py`, `src/ragnarok/detection/rfdetr_torch.py`, `src/ragnarok/detection/factory.py`, `tests/detection/__init__.py`, `tests/detection/test_base.py`, `tests/detection/test_rfdetr_torch.py`

**Interfaces:**
- Consumes: `Frame`, `Detection`, `Detections` (Task 3), `DetectionConfig` (Task 4).
- Produces:
  - `Detector` ABC: `detect(frame: Frame) -> Detections`.
  - `to_detections(sv_detections) -> Detections` — maps a `supervision.Detections` (`.xyxy`, `.confidence`, `.class_id`) to our `Detections`.
  - `RFDETRTorchDetector(config, *, model=None)` — model injectable (a fake with `.predict(image, threshold) -> sv.Detections`).
  - `create_detector(config: DetectionConfig) -> Detector`.

- [ ] **Step 1: Write the failing test (mapping + injected fake model)**

```python
# tests/detection/test_base.py
import numpy as np
from types import SimpleNamespace
from ragnarok.detection.base import to_detections

def test_to_detections_maps_fields():
    sv = SimpleNamespace(
        xyxy=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
    )
    dets = to_detections(sv)
    assert len(dets) == 2
    assert list(dets)[0].xyxy == (1.0, 2.0, 3.0, 4.0)
    assert list(dets)[1].confidence == 0.8

def test_to_detections_empty():
    sv = SimpleNamespace(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))
    assert len(to_detections(sv)) == 0
```

```python
# tests/detection/test_rfdetr_torch.py
import numpy as np
from types import SimpleNamespace
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector
from ragnarok.core.types import Frame

class _FakeModel:
    def __init__(self): self.threshold = None
    def predict(self, image, threshold=0.5):
        self.threshold = threshold
        return SimpleNamespace(xyxy=np.array([[10.0, 10.0, 20.0, 30.0]]),
                               confidence=np.array([0.95]), class_id=np.array([0]))

def test_detect_returns_detections_and_passes_threshold():
    det = RFDETRTorchDetector(DetectionConfig(confidence=0.6), model=_FakeModel())
    frame = Frame(image=np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))
    out = det.detect(frame)
    assert len(out) == 1
    assert list(out)[0].xyxy == (10.0, 10.0, 20.0, 30.0)
    assert det._model.threshold == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/detection/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.detection'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/detection/__init__.py
```

```python
# src/ragnarok/detection/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from ragnarok.core.types import Frame, Detection, Detections

def to_detections(sv) -> Detections:
    items = tuple(
        Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)),
                  confidence=float(c), class_id=int(k))
        for (x1, y1, x2, y2), c, k in zip(sv.xyxy, sv.confidence, sv.class_id)
    )
    return Detections(items=items)

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> Detections: ...
```

```python
# src/ragnarok/detection/rfdetr_torch.py
from __future__ import annotations
import cv2
from ragnarok.core.types import Frame, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector, to_detections

_MODEL_CLASSES = {
    "nano": "RFDETRNano", "small": "RFDETRSmall",
    "medium": "RFDETRMedium", "large": "RFDETRLarge",
}

class RFDETRTorchDetector(Detector):
    def __init__(self, config: DetectionConfig, *, model=None) -> None:
        self._config = config
        if model is None:
            import rfdetr  # lazy: keeps torch/weights out of unit tests
            model = getattr(rfdetr, _MODEL_CLASSES[config.model])()
        self._model = model

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        sv = self._model.predict(rgb, threshold=self._config.confidence)
        return to_detections(sv)
```

```python
# src/ragnarok/detection/factory.py
from __future__ import annotations
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector

def create_detector(config: DetectionConfig) -> Detector:
    from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector
    return RFDETRTorchDetector(config)
```

```python
# tests/detection/__init__.py
```

Note: add `opencv-python>=4.9` to `pyproject.toml` `dependencies` (used here and in later phases). Re-run `pip install -e ".[dev]"` after editing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/detection/ -v`
Expected: PASS

- [ ] **Step 5: Manual smoke (real RF-DETR weights; not in CI)**

On the box with `rfdetr`+torch installed:
`python -c "import numpy as np; from ragnarok.config.schema import DetectionConfig; from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector; d=RFDETRTorchDetector(DetectionConfig()); from ragnarok.core.types import Frame; print(len(d.detect(Frame(np.zeros((384,384,3),np.uint8),1,(0,0,384,384)))))"`
Expected: prints `0` (no objects in a black frame) without error — confirms weights load and the pipeline runs.

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/detection/ tests/detection/ pyproject.toml
git commit -m "feat(detection): add Detector interface and RF-DETR torch backend"
```

---

### Task 11: Worker loop

**Files:**
- Create: `src/ragnarok/worker/__init__.py`, `src/ragnarok/worker/loop.py`, `tests/worker/__init__.py`, `tests/worker/test_loop.py`

**Interfaces:**
- Consumes: `Capturer` (Task 8), `Detector` (Task 10), `StageProfiler` (Task 6), `SnapshotPublisher`/`TelemetrySnapshot` (Task 7), `now_ns` (Task 2).
- Produces:
  - `WorkerLoop(capturer, detector, profiler, publisher, *, preview_max=320)` — `tick() -> None` (one capture→detect→profile→publish iteration); `run(stop_event) -> None` (loop until set); `stop()` helper. Each `tick` records `"capture"`, `"infer"`, `"loop"` stages and publishes an incrementing-`seq` snapshot.

- [ ] **Step 1: Write the failing test (fakes; no GPU/display)**

```python
# tests/worker/test_loop.py
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop

class _Cap:
    def start(self): ...
    def grab(self):
        return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))
    def stop(self): ...

class _Det:
    def detect(self, frame):
        return Detections(items=(Detection((0, 0, 10, 10), 0.9, 0),))

def test_tick_publishes_snapshot_with_detection_count():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.tick()
    snap = pub.latest()
    assert snap is not None
    assert snap.detection_count == 1
    assert snap.seq == 1
    assert snap.preview is not None  # downscaled frame attached

def test_tick_skips_publish_on_no_frame():
    class _NoCap(_Cap):
        def grab(self): return None
    pub = SnapshotPublisher()
    loop = WorkerLoop(_NoCap(), _Det(), StageProfiler(), pub)
    loop.tick()
    assert pub.latest() is None

def test_seq_increments():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.tick(); loop.tick()
    assert pub.latest().seq == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worker/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.worker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/worker/__init__.py
```

```python
# src/ragnarok/worker/loop.py
from __future__ import annotations
import threading
import cv2
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher

class WorkerLoop:
    def __init__(self, capturer, detector, profiler: StageProfiler,
                 publisher: SnapshotPublisher, *, preview_max: int = 320) -> None:
        self._cap = capturer
        self._det = detector
        self._profiler = profiler
        self._pub = publisher
        self._preview_max = preview_max
        self._seq = 0
        self._last_ns: int | None = None

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, self._preview_max / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return np.ascontiguousarray(image)

    def tick(self) -> None:
        t0 = now_ns()
        frame = self._cap.grab()
        t_cap = now_ns()
        if frame is None:
            return
        dets = self._det.detect(frame)
        t_inf = now_ns()

        self._profiler.record("capture", t_cap - t0)
        self._profiler.record("infer", t_inf - t_cap)
        self._profiler.record("loop", t_inf - t0)

        fps = 0.0
        if self._last_ns is not None:
            dt = t_inf - self._last_ns
            fps = 1e9 / dt if dt > 0 else 0.0
        self._last_ns = t_inf

        p50, p99 = self._profiler.percentiles("loop")
        self._seq += 1
        self._pub.publish(TelemetrySnapshot(
            fps=fps, loop_ms_p50=p50, loop_ms_p99=p99,
            detection_count=len(dets), preview=self._downscale(frame.image), seq=self._seq,
        ))

    def run(self, stop_event: threading.Event) -> None:
        self._cap.start()
        try:
            while not stop_event.is_set():
                self.tick()
        finally:
            self._cap.stop()
```

```python
# tests/worker/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/worker/test_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/worker/ tests/worker/
git commit -m "feat(worker): add capture->detect->profile->publish loop"
```

---

### Task 12: GUI shell + thread + entry point

**Files:**
- Create: `src/ragnarok/gui/__init__.py`, `src/ragnarok/gui/worker_thread.py`, `src/ragnarok/gui/main_window.py`, `src/ragnarok/app.py`, `tests/gui/__init__.py`, `tests/gui/test_main_window.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry `ragnarok = "ragnarok.app:main"`)

**Interfaces:**
- Consumes: `WorkerLoop` (Task 11), `SnapshotPublisher`/`TelemetrySnapshot` (Task 7), `ConfigHandle`/`load_config` (Task 5), `create_capturer`/`create_detector` factories.
- Produces:
  - `WorkerThread(loop)` — `QThread` subclass; `run()` calls `loop.run(self._stop)`; `stop()` sets the event and waits.
  - `MainWindow(publisher)` — polls `publisher.latest()` on a 60 Hz `QTimer`, renders preview (`QImage` Format_BGR888) + an FPS / p50 / p99 / detections label.
  - `main() -> int` — composition root.

- [ ] **Step 1: Write the failing test (pytest-qt; no GPU/real capture)**

```python
# tests/gui/test_main_window.py
import numpy as np
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.gui.main_window import MainWindow

def test_window_renders_latest_snapshot(qtbot):
    pub = SnapshotPublisher()
    win = MainWindow(pub)
    qtbot.addWidget(win)
    pub.publish(TelemetrySnapshot(
        fps=123.4, loop_ms_p50=5.0, loop_ms_p99=9.0, detection_count=2,
        preview=np.zeros((100, 100, 3), np.uint8), seq=1))
    win.refresh()  # the QTimer slot, called directly
    assert "123.4" in win.stats_label.text()
    assert "2" in win.stats_label.text()

def test_window_handles_no_snapshot(qtbot):
    win = MainWindow(SnapshotPublisher())
    qtbot.addWidget(win)
    win.refresh()  # must not raise when latest() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gui/test_main_window.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragnarok.gui'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/gui/__init__.py
```

```python
# src/ragnarok/gui/worker_thread.py
from __future__ import annotations
import threading
from PySide6.QtCore import QThread

class WorkerThread(QThread):
    def __init__(self, loop) -> None:
        super().__init__()
        self._loop = loop
        self._stop = threading.Event()

    def run(self) -> None:  # executes in the new thread
        self._loop.run(self._stop)

    def stop(self) -> None:
        self._stop.set()
        self.wait(2000)
```

```python
# src/ragnarok/gui/main_window.py
from __future__ import annotations
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QMainWindow
from ragnarok.telemetry.snapshot import SnapshotPublisher

class MainWindow(QMainWindow):
    def __init__(self, publisher: SnapshotPublisher) -> None:
        super().__init__()
        self.setWindowTitle("Ragnarok")
        self._pub = publisher
        central = QWidget()
        layout = QVBoxLayout(central)
        self.preview_label = QLabel("no signal")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.stats_label = QLabel("--")
        layout.addWidget(self.preview_label)
        layout.addWidget(self.stats_label)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    def refresh(self) -> None:
        snap = self._pub.latest()
        if snap is None:
            return
        self.stats_label.setText(
            f"FPS {snap.fps:.1f}  |  loop p50 {snap.loop_ms_p50:.1f} ms  "
            f"p99 {snap.loop_ms_p99:.1f} ms  |  detections {snap.detection_count}"
        )
        if snap.preview is not None:
            img = snap.preview
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))
```

```python
# src/ragnarok/app.py
from __future__ import annotations
import sys
from pathlib import Path
import os
from PySide6.QtWidgets import QApplication
from ragnarok.config.store import load_config
from ragnarok.capture.factory import create_capturer
from ragnarok.detection.factory import create_detector
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop
from ragnarok.gui.worker_thread import WorkerThread
from ragnarok.gui.main_window import MainWindow

def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Ragnarok"
    return base / "config.toml"

def main() -> int:
    app = QApplication(sys.argv)
    cfg = load_config(_config_path())
    publisher = SnapshotPublisher()
    loop = WorkerLoop(
        create_capturer(cfg.capture), create_detector(cfg.detection),
        StageProfiler(), publisher,
    )
    worker = WorkerThread(loop)
    window = MainWindow(publisher)
    window.show()
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# tests/gui/__init__.py
```

Add to `pyproject.toml`:
```toml
[project.scripts]
ragnarok = "ragnarok.app:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gui/test_main_window.py -v`
Expected: PASS (pytest-qt provides the `qtbot` fixture and a QApplication).

- [ ] **Step 5: Manual end-to-end smoke (real, on the Windows box)**

Run: `ragnarok` (or `python -m ragnarok.app`)
Expected: a window opens; with the game (or any window) visible it shows a live centered-ROI preview and a non-zero FPS with p50/p99 latency. Detections will be 0 until a model is fine-tuned (Phase 6) — that's expected.

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/gui/ src/ragnarok/app.py tests/gui/ pyproject.toml
git commit -m "feat(gui): add worker thread, main window shell, entry point"
```

---

## Self-Review

**1. Spec coverage (Phase 1 scope, spec §17.1):**
- Project skeleton → Task 1. ✓
- `perf_counter_ns` timestamping → Task 2 + stamped in `Frame` at capture (Tasks 8/9), propagated to the loop (Task 11). ✓
- Capture (BetterCam ROI default + mss fallback, `Capturer` interface, centered ROI) → Tasks 8/9. ✓
- RF-DETR detection (Torch baseline, Apache-2.0 variants, confidence threshold, `Detector` interface) → Task 10. ✓ (TensorRT engine + dynamic-ROI explicitly deferred to Phase 5 per scope note.)
- Per-stage latency profiler (p50/p99) → Task 6, wired in Task 11. ✓
- Telemetry snapshot plumbing (publish-latest, GIL-safe, GUI polls ≤60 Hz, no hot-path signals) → Tasks 7/11/12. ✓
- Config (pydantic + TOML + atomic snapshot) → Tasks 4/5. ✓
- GUI shell (preview + FPS/latency, worker off GUI thread) → Task 12. ✓
- Two-thread/one-process, CUDA only in worker → Tasks 11/12 (worker thread owns capture+detect). ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to"; every code step has runnable code; hardware/weights/GUI integration that can't be unit-tested is covered by injected fakes plus clearly-marked manual smoke steps. ✓

**3. Type consistency:** `Frame(image, t_capture_ns, region)`, `Detection(xyxy, confidence, class_id)`, `Detections(items=...)`, `Capturer.grab()->Frame|None`, `Detector.detect(frame)->Detections`, `to_detections(sv)`, `TelemetrySnapshot(fps, loop_ms_p50, loop_ms_p99, detection_count, preview, seq)`, `SnapshotPublisher.publish/latest`, `WorkerLoop(capturer, detector, profiler, publisher)`, `ConfigHandle.current/.swap` — names/signatures used in Tasks 11–12 match their definitions in Tasks 3–10. ✓

Deferred-by-design (not gaps; later phases): tracking, friend/foe, targeting/filtering/aim, output drivers + firmware, recoil, trigger, training pipeline, overlay, diagnostics, TensorRT/dynamic-ROI, world-space ego-motion. Each gets its own phase plan.
