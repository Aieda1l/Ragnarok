# Ragnarok Phase 6A — Capture & Benchmark CI-Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CI-safe core of the training pipeline (spec §12): an in-app smart-sampling **frame grabber** to collect game frames worth labeling, a pure **benchmark harness** (mAP@0.75 + center-error + latency) to measure any detector on a labeled frame set, the **hard-example** selection policy, and a `TrainingConfig` to drive them.

**Architecture:** A new `src/ragnarok/training/` package of small, pure, dependency-injected units (matching Phases 1–5): the sampling *decision* and scene-change metric are pure functions; the `FrameGrabber` rate-limits and delegates disk writes to an injected writer (so CI never touches disk); the benchmark metrics (IoU, VOC all-point AP@0.75, center-error) are pure numpy and tested against analytically-known cases; the benchmark runner evaluates any `Detector` over an injected `(Frame, gt_boxes)` set with an injected clock. The actual Roboflow upload/annotation and ONNX/TRT export are **separate later plans (6B/6C)** — this plan deliberately ships only what is unit-testable now and immediately unblocks data collection + detector measurement.

**Tech Stack:** Python 3.11+, numpy, OpenCV (already used), pydantic/tomlkit config. No new third-party dependencies (the `roboflow`/network and `trtexec`/GPU dependencies belong to 6B/6C).

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is the CV training/eval tooling (spec §12, §15).
- **No game ground truth** — the dataset comes from Roboflow; the benchmark consumes a *labeled* frame set (gt boxes the user annotated), not engine internals (spec §1 non-goals, §12).
- **CI-safe always:** no GPU/display/disk/network in unit tests. The frame grabber's disk write is an injected callable; the benchmark runs against a fake `Detector` + synthetic `(Frame, gt)` set with an injected clock; all metrics are pure numpy. Modules import without torch/rfdetr/cv2.cuda.
- **Integer-ns time math** (`perf_counter_ns`); float-seconds only at boundaries.
- **No secrets in committed config:** the Roboflow API key is NOT a config field — it is read from the `RAGNAROK_ROBOFLOW_API_KEY` environment variable by the 6B client. `TrainingConfig` holds only non-secret workspace/project/version + paths.
- **Frozen pydantic config**, backward-compatible TOML round-trip (existing `config.toml` without `[training]` still loads).
- **Single `player` class** (spec §5.2): the benchmark is single-class (`map@0.75` == AP for the one class).
- **TDD, frequent commits, exact file paths.** Match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, module docstrings, focused files).

## Scope Boundary (explicit deferrals — Plans 6B / 6C / later)

- **Roboflow client** (upload frames, trigger COCO export, download dataset, push hard examples) → **Plan 6B**. Needs network + an API key; the client logic will use an injected transport, but it's its own subsystem. This plan ships the *hard-example selection policy* (pure) that 6B will consume.
- **ONNX / TensorRT export + engine detector backend** (`RFDETROnnx`/`RFDETRTensorRT`, `trtexec`, INT8/sparsity) → **Plan 6C**. Mostly box-only (real export on the 3090). This plan ships the benchmark harness 6C will use to compare FP16 vs INT8.
- **Actual `rfdetr train` run** → box-only (the user's GPU + a labeled Roboflow dataset). Out of scope for all CI plans.
- **Live disk capture during real gameplay** → box-only smoke (the `FrameGrabber` is unit-tested with an injected writer; wiring it into the live worker is a small follow-up once the user wants to collect).

---

## File Structure

**New files:**
- `src/ragnarok/training/__init__.py` — package marker.
- `src/ragnarok/training/metrics.py` — pure detection metrics: `iou`, `average_precision_at_iou`, `center_error`.
- `src/ragnarok/training/benchmark.py` — `BenchmarkResult` + `run_benchmark`.
- `src/ragnarok/training/sampling.py` — pure capture-decision: `scene_change_fraction`, `should_capture`.
- `src/ragnarok/training/grabber.py` — `FrameGrabber` (rate-limited, injected writer).
- `src/ragnarok/training/hard_examples.py` — `select_hard_examples` (pure).
- `tests/training/__init__.py` + one test module per source module.

**Modified files:**
- `src/ragnarok/config/schema.py` — add `TrainingConfig`, nest in `AppConfig`.
- `tests/config/test_training_config.py` (new).

---

## Task 1: TrainingConfig

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Test: `tests/config/test_training_config.py`

**Interfaces:**
- Produces: `TrainingConfig` (frozen) with `frames_dir: str = "captures"`, `dataset_dir: str = "datasets"`, `engines_dir: str = "engines"`, `roboflow_workspace: str = ""`, `roboflow_project: str = ""`, `roboflow_version: int = Field(default=1, ge=1)`, `capture_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)`, `scene_change_threshold: float = Field(default=0.15, ge=0.0, le=1.0)`, `min_capture_interval_s: float = Field(default=0.5, ge=0.0)`, `hard_example_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)`. Nested as `AppConfig.training`. NO api-key field (env var only — documented in the docstring).

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_training_config.py
"""Tests for TrainingConfig + its nesting in AppConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import TrainingConfig, AppConfig


def test_defaults():
    t = TrainingConfig()
    assert t.frames_dir == "captures"
    assert t.dataset_dir == "datasets"
    assert t.engines_dir == "engines"
    assert t.roboflow_workspace == "" and t.roboflow_project == ""
    assert t.roboflow_version == 1
    assert t.capture_conf_threshold == 0.5
    assert t.scene_change_threshold == 0.15
    assert t.min_capture_interval_s == 0.5
    assert t.hard_example_conf_threshold == 0.5


def test_no_api_key_field():
    # The Roboflow API key must NOT be a config field (env var only).
    assert "api_key" not in TrainingConfig.model_fields
    assert "roboflow_api_key" not in TrainingConfig.model_fields


def test_bounds():
    with pytest.raises(ValidationError):
        TrainingConfig(capture_conf_threshold=1.5)
    with pytest.raises(ValidationError):
        TrainingConfig(roboflow_version=0)


def test_nested_and_backward_compatible():
    assert isinstance(AppConfig().training, TrainingConfig)
    app = AppConfig(detection={"model": "nano"})       # no [training] section
    assert app.training.frames_dir == "captures"


def test_frozen():
    with pytest.raises(Exception):
        TrainingConfig().frames_dir = "x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_training_config.py -v`
Expected: FAIL — `cannot import name 'TrainingConfig'`.

- [ ] **Step 3: Add TrainingConfig and nest it**

In `src/ragnarok/config/schema.py`, add (before `AppConfig`):

```python
class TrainingConfig(BaseModel):
    """Training-pipeline config (spec §12).

    NOTE: the Roboflow API key is intentionally NOT a field here — it is read
    from the RAGNAROK_ROBOFLOW_API_KEY environment variable by the Roboflow
    client (Plan 6B), so secrets never land in a committed/example config.
    Paths are relative to the Ragnarok app data dir.
    """
    model_config = ConfigDict(frozen=True)
    frames_dir: str = "captures"
    dataset_dir: str = "datasets"
    engines_dir: str = "engines"
    roboflow_workspace: str = ""
    roboflow_project: str = ""
    roboflow_version: int = Field(default=1, ge=1)
    capture_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    scene_change_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    min_capture_interval_s: float = Field(default=0.5, ge=0.0)
    hard_example_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
```

Add to `AppConfig` (after `diagnostics`):

```python
    training: TrainingConfig = TrainingConfig()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config -q`
Expected: PASS (incl. TOML round-trip in `test_store.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py tests/config/test_training_config.py
git commit -m "feat(config): TrainingConfig (capture/dataset/engine paths + sampling thresholds; api key via env)"
```

---

## Task 2: Detection metrics (IoU, AP@0.75, center-error) — pure

**Files:**
- Create: `src/ragnarok/training/__init__.py` (empty), `src/ragnarok/training/metrics.py`
- Create: `tests/training/__init__.py` (empty), `tests/training/test_metrics.py`

**Interfaces:**
- Consumes: nothing (numpy).
- Produces:
  - `iou(box_a, box_b) -> float` — IoU of two `(x1,y1,x2,y2)` boxes; 0.0 on no overlap or degenerate box.
  - `average_precision_at_iou(preds, gts, *, iou_thresh=0.75) -> float` — single-class VOC all-point AP. `preds` = list of `(xyxy, score)`; `gts` = list of `xyxy`. Greedy highest-score-first matching to unmatched gt with IoU ≥ thresh. Conventions: no gt and no preds → 1.0; no gt but preds → 0.0; gts but no preds → 0.0.
  - `center_error(preds, gts, *, iou_thresh=0.5) -> float | None` — mean Euclidean distance between matched (IoU ≥ thresh) predicted/gt box centers; `None` if no matches. `preds` here may be plain `xyxy` list or `(xyxy, score)` (accept both by reading index 0 when it's a pair).

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_metrics.py
"""Tests for pure detection metrics against analytic cases."""
from __future__ import annotations
import math
from ragnarok.training.metrics import iou, average_precision_at_iou, center_error


def test_iou_half_overlap():
    a = (0.0, 0.0, 2.0, 2.0)        # area 4
    b = (1.0, 0.0, 3.0, 2.0)        # area 4, overlap x in [1,2] -> 2*2=2... overlap area = 1*2 = 2
    # intersection = 2, union = 4+4-2 = 6 -> 1/3
    assert abs(iou(a, b) - (2.0 / 6.0)) < 1e-9


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_ap_perfect_detection():
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9)]   # exact match, IoU 1.0
    assert average_precision_at_iou(preds, gts, iou_thresh=0.75) == 1.0


def test_ap_low_iou_is_miss():
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((5.0, 5.0, 15.0, 15.0), 0.9)]   # IoU well below 0.75
    assert average_precision_at_iou(preds, gts, iou_thresh=0.75) == 0.0


def test_ap_half_recall_is_half():
    # two gt, one matched -> recall caps at 0.5, precision 1.0 -> AP 0.5
    gts = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9)]
    assert abs(average_precision_at_iou(preds, gts, iou_thresh=0.75) - 0.5) < 1e-9


def test_ap_extra_false_positive_does_not_reduce_perfect_recall():
    # one tp (high score) + one fp (low score) vs one gt -> envelope keeps AP 1.0
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9), ((50.0, 50.0, 60.0, 60.0), 0.1)]
    assert abs(average_precision_at_iou(preds, gts, iou_thresh=0.75) - 1.0) < 1e-9


def test_ap_no_gt_no_preds_is_one():
    assert average_precision_at_iou([], [], iou_thresh=0.75) == 1.0


def test_ap_no_gt_with_preds_is_zero():
    assert average_precision_at_iou([((0, 0, 1, 1), 0.9)], [], iou_thresh=0.75) == 0.0


def test_center_error_matched_distance():
    gts = [(0.0, 0.0, 10.0, 10.0)]                 # center (5,5)
    preds = [(0.0, 2.0, 10.0, 12.0)]               # center (5,7), IoU 0.667 >= 0.5 -> matches; dist 2
    assert abs(center_error(preds, gts) - 2.0) < 1e-9


def test_center_error_none_when_no_match():
    assert center_error([(0, 0, 1, 1)], [(100, 100, 110, 110)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_metrics.py -v`
Expected: FAIL — `No module named 'ragnarok.training'`.

- [ ] **Step 3: Implement metrics.py**

```python
# src/ragnarok/training/__init__.py
```

```python
# src/ragnarok/training/metrics.py
"""Pure single-class detection metrics (spec §12 step 5, §15).

IoU, VOC all-point Average Precision at a fixed IoU threshold (mAP@0.75 for the
single 'player' class == AP), and mean matched center-error. All pure numpy,
tested against analytically-known cases — no engine/GPU/frames.
"""
from __future__ import annotations

import math

import numpy as np


def _xyxy(box):
    # accept either a plain xyxy or a (xyxy, score) pair
    if len(box) == 2 and hasattr(box[0], "__len__"):
        return box[0]
    return box


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def average_precision_at_iou(preds, gts, *, iou_thresh: float = 0.75) -> float:
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    matched = [False] * len(gts)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for rank, i in enumerate(order):
        box = preds[i][0]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched[best_j] = True
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # VOC all-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_error(preds, gts, *, iou_thresh: float = 0.5) -> float | None:
    pred_boxes = [_xyxy(p) for p in preds]
    matched = [False] * len(pred_boxes)
    dists: list[float] = []
    for g in gts:
        best_iou, best_i = 0.0, -1
        for i, p in enumerate(pred_boxes):
            if matched[i]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_i >= 0 and best_iou >= iou_thresh:
            matched[best_i] = True
            gc, pc = _center(g), _center(pred_boxes[best_i])
            dists.append(math.hypot(gc[0] - pc[0], gc[1] - pc[1]))
    if not dists:
        return None
    return float(sum(dists) / len(dists))
```

```python
# tests/training/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/__init__.py src/ragnarok/training/metrics.py tests/training
git commit -m "feat(training): pure detection metrics (IoU, VOC AP@0.75, center-error)"
```

---

## Task 3: Benchmark harness (BenchmarkResult + run_benchmark)

**Files:**
- Create: `src/ragnarok/training/benchmark.py`
- Create: `tests/training/test_benchmark.py`

**Interfaces:**
- Consumes: `average_precision_at_iou`, `center_error` (T2); `Detector` ABC (`ragnarok.detection.base`); `now_ns` (`ragnarok.core.clock`).
- Produces:
  - `@dataclass(frozen=True) BenchmarkResult` with `map75: float`, `center_error: float | None`, `latency_p50_ms: float`, `latency_p99_ms: float`, `n_images: int`.
  - `run_benchmark(detector, dataset, *, clock=now_ns, iou_thresh=0.75) -> BenchmarkResult` — `dataset` is a list of `(Frame, gt_boxes)` where `gt_boxes` is a list of `xyxy`. For each item: time `detector.detect(frame)` with `clock()` (ns), collect `(xyxy, confidence)` preds; aggregate AP@iou_thresh and center-error across ALL images (concatenated), and latency p50/p99 in ms. CI-safe with a fake `Detector` + a fake clock + synthetic `Frame`s.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_benchmark.py
"""Tests for the benchmark harness against a fake detector (no GPU/frames)."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.training.benchmark import run_benchmark, BenchmarkResult


def _frame():
    return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))


class _PerfectDetector:
    """Returns exactly the gt box (passed via a side channel) with conf 0.9."""
    def __init__(self, boxes):
        self._boxes = boxes
    def detect(self, frame):
        return Detections(items=tuple(
            Detection(xyxy=b, confidence=0.9, class_id=0) for b in self._boxes))


class _StepClock:
    def __init__(self, dt_ns):
        self.t = 0
        self._dt = dt_ns
    def __call__(self):
        v = self.t
        self.t += self._dt
        return v


def test_perfect_detector_scores_map_1():
    gt = [(0.0, 0.0, 10.0, 10.0)]
    dataset = [(_frame(), gt)]
    res = run_benchmark(_PerfectDetector(gt), dataset, clock=_StepClock(2_000_000))
    assert isinstance(res, BenchmarkResult)
    assert res.map75 == 1.0
    assert res.center_error == 0.0
    assert res.n_images == 1
    assert res.latency_p50_ms > 0.0


def test_aggregates_across_images():
    g1 = [(0.0, 0.0, 10.0, 10.0)]
    g2 = [(20.0, 20.0, 30.0, 30.0)]
    dataset = [(_frame(), g1), (_frame(), g2)]
    # detector returns g1 for both images -> image 2 is a miss
    res = run_benchmark(_PerfectDetector(g1), dataset, clock=_StepClock(1_000_000))
    assert res.n_images == 2
    assert 0.0 <= res.map75 <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_benchmark.py -v`
Expected: FAIL — `No module named 'ragnarok.training.benchmark'`.

- [ ] **Step 3: Implement benchmark.py**

```python
# src/ragnarok/training/benchmark.py
"""Benchmark harness: measure a Detector on a labeled frame set (spec §12.5, §15).

Aggregates single-class AP@iou_thresh + matched center-error across all images,
and reports latency p50/p99. CI-safe: inject a fake Detector + clock + synthetic
frames; real engines over real frames are a box-only smoke.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragnarok.core.clock import now_ns
from ragnarok.training.metrics import average_precision_at_iou, center_error


@dataclass(frozen=True)
class BenchmarkResult:
    map75: float
    center_error: float | None
    latency_p50_ms: float
    latency_p99_ms: float
    n_images: int


def run_benchmark(detector, dataset, *, clock=now_ns, iou_thresh: float = 0.75) -> BenchmarkResult:
    all_preds: list[tuple[tuple, float]] = []
    all_gts: list[tuple] = []
    latencies_ns: list[int] = []
    for frame, gt_boxes in dataset:
        t0 = clock()
        dets = detector.detect(frame)
        t1 = clock()
        latencies_ns.append(int(t1 - t0))
        all_preds.extend((d.xyxy, d.confidence) for d in dets)
        all_gts.extend(tuple(b) for b in gt_boxes)
    map75 = average_precision_at_iou(all_preds, all_gts, iou_thresh=iou_thresh)
    cerr = center_error(all_preds, all_gts)
    if latencies_ns:
        arr = np.asarray(latencies_ns, dtype=np.int64)
        p50, p99 = (float(v) / 1e6 for v in np.percentile(arr, [50, 99]))
    else:
        p50 = p99 = 0.0
    return BenchmarkResult(map75=map75, center_error=cerr,
                           latency_p50_ms=p50, latency_p99_ms=p99,
                           n_images=len(dataset))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/benchmark.py tests/training/test_benchmark.py
git commit -m "feat(training): benchmark harness (mAP@0.75 + center-error + latency over a labeled set)"
```

---

## Task 4: Smart-sampling decision (scene-change + should_capture) — pure

**Files:**
- Create: `src/ragnarok/training/sampling.py`
- Create: `tests/training/test_sampling.py`

**Interfaces:**
- Consumes: nothing (numpy); reads `Detections` (iterable of `Detection` with `.confidence`).
- Produces:
  - `scene_change_fraction(img_a, img_b) -> float` — mean absolute per-pixel difference of two same-shape uint8 BGR images, normalized to `[0, 1]` (divide by 255). Returns `1.0` if shapes differ (treat as a full scene change).
  - `should_capture(detections, frame_image, last_saved_image, *, conf_threshold, scene_change_threshold) -> bool` — `True` when the frame is worth labeling: **uncertain** (no detections, or max detection confidence `< conf_threshold`) OR a **scene change** (`last_saved_image is None`, or `scene_change_fraction(frame_image, last_saved_image) >= scene_change_threshold`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_sampling.py
"""Tests for the pure smart-sampling capture decision."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Detection, Detections
from ragnarok.training.sampling import scene_change_fraction, should_capture


def _img(val):
    return np.full((8, 8, 3), val, np.uint8)


def test_scene_change_identical_is_zero():
    assert scene_change_fraction(_img(100), _img(100)) == 0.0


def test_scene_change_full_swing_is_one():
    assert abs(scene_change_fraction(_img(0), _img(255)) - 1.0) < 1e-6


def test_scene_change_shape_mismatch_is_one():
    assert scene_change_fraction(_img(0), np.zeros((4, 4, 3), np.uint8)) == 1.0


def test_capture_on_low_confidence():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.2, 0),))   # below threshold
    assert should_capture(dets, _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_capture_on_no_detections():
    assert should_capture(Detections.empty(), _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_capture_on_scene_change_even_when_confident():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(0), _img(255),
                          conf_threshold=0.5, scene_change_threshold=0.15) is True


def test_no_capture_when_confident_and_static():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(100), _img(100),
                          conf_threshold=0.5, scene_change_threshold=0.15) is False


def test_capture_when_no_last_saved():
    dets = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    assert should_capture(dets, _img(100), None,
                          conf_threshold=0.5, scene_change_threshold=0.15) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_sampling.py -v`
Expected: FAIL — `No module named 'ragnarok.training.sampling'`.

- [ ] **Step 3: Implement sampling.py**

```python
# src/ragnarok/training/sampling.py
"""Smart-sampling capture decision for the frame grabber (spec §12 step 1).

Pure functions: keep a frame for labeling when the detector is UNCERTAIN about
it (no/low-confidence detections — the hard examples worth labeling) or when the
SCENE CHANGED vs the last saved frame (coverage/diversity). No IO here.
"""
from __future__ import annotations

import numpy as np


def scene_change_fraction(img_a, img_b) -> float:
    a = np.asarray(img_a)
    b = np.asarray(img_b)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)


def should_capture(detections, frame_image, last_saved_image, *,
                   conf_threshold: float, scene_change_threshold: float) -> bool:
    confs = [d.confidence for d in detections]
    uncertain = (not confs) or (max(confs) < conf_threshold)
    if uncertain:
        return True
    if last_saved_image is None:
        return True
    return scene_change_fraction(frame_image, last_saved_image) >= scene_change_threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/sampling.py tests/training/test_sampling.py
git commit -m "feat(training): pure smart-sampling capture decision (uncertainty + scene-change)"
```

---

## Task 5: FrameGrabber (rate-limited, injected writer)

**Files:**
- Create: `src/ragnarok/training/grabber.py`
- Create: `tests/training/test_grabber.py`

**Interfaces:**
- Consumes: `should_capture` (T4); `now_ns` (`ragnarok.core.clock`); `Frame`/`Detections` (`ragnarok.core.types`).
- Produces: `FrameGrabber(*, writer, conf_threshold, scene_change_threshold, min_interval_s, clock=now_ns)`. `writer` is a callable `(image: np.ndarray, t_capture_ns: int) -> None` (disk IO injected — CI uses a recorder). `.offer(frame, detections) -> bool` — applies the min-interval rate limit AND `should_capture`; on accept, calls `writer(frame.image, frame.t_capture_ns)`, updates the last-saved image + last-save time, returns `True`; else returns `False`. `.count` property = number saved.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_grabber.py
"""Tests for the FrameGrabber (rate-limit + injected writer; no disk)."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.training.grabber import FrameGrabber


def _frame(val, t_ns):
    return Frame(np.full((8, 8, 3), val, np.uint8), t_capture_ns=t_ns, region=(0, 0, 8, 8))


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def _uncertain():
    return Detections.empty()              # no detections -> always "worth capturing"


def test_saves_first_uncertain_frame():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append((img.copy(), t)),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.5, clock=clk)
    assert g.offer(_frame(100, 1), _uncertain()) is True
    assert g.count == 1 and len(saved) == 1 and saved[0][1] == 1


def test_rate_limit_blocks_within_interval():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append(t),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.5, clock=clk)
    g.offer(_frame(100, 1), _uncertain())          # saved at t=0
    clk.t = 200_000_000                            # 200 ms < 500 ms
    assert g.offer(_frame(0, 2), _uncertain()) is False
    clk.t = 600_000_000                            # 600 ms >= 500 ms
    assert g.offer(_frame(0, 3), _uncertain()) is True
    assert g.count == 2


def test_skips_confident_static_frames():
    saved = []
    clk = _Clock()
    g = FrameGrabber(writer=lambda img, t: saved.append(t),
                     conf_threshold=0.5, scene_change_threshold=0.15,
                     min_interval_s=0.0, clock=clk)
    confident = Detections(items=(Detection((0, 0, 1, 1), 0.99, 0),))
    g.offer(_frame(100, 1), confident)             # first -> last_saved None -> saved
    assert g.count == 1
    clk.t = 1_000_000_000
    assert g.offer(_frame(100, 2), confident) is False   # confident + identical -> skip
    assert g.count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_grabber.py -v`
Expected: FAIL — `No module named 'ragnarok.training.grabber'`.

- [ ] **Step 3: Implement grabber.py**

```python
# src/ragnarok/training/grabber.py
"""In-app smart frame grabber (spec §12 step 1).

Rate-limited; delegates the actual disk write to an injected `writer` callable so
the decision logic is unit-testable and CI never touches disk. Saves frames the
detector is uncertain about or that show a scene change (see sampling.py).
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.training.sampling import should_capture


class FrameGrabber:
    def __init__(self, *, writer, conf_threshold: float, scene_change_threshold: float,
                 min_interval_s: float, clock=now_ns) -> None:
        self._writer = writer
        self._conf = conf_threshold
        self._scene = scene_change_threshold
        self._min_interval_ns = int(min_interval_s * 1e9)
        self._clock = clock
        self._last_saved_image = None
        self._last_save_ns: int | None = None
        self.count = 0

    def offer(self, frame, detections) -> bool:
        now = self._clock()
        if self._last_save_ns is not None and now - self._last_save_ns < self._min_interval_ns:
            return False
        if not should_capture(detections, frame.image, self._last_saved_image,
                              conf_threshold=self._conf,
                              scene_change_threshold=self._scene):
            return False
        self._writer(frame.image, frame.t_capture_ns)
        self._last_saved_image = frame.image
        self._last_save_ns = now
        self.count += 1
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_grabber.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/grabber.py tests/training/test_grabber.py
git commit -m "feat(training): FrameGrabber (rate-limited smart sampling, injected writer)"
```

---

## Task 6: Hard-example selection — pure

**Files:**
- Create: `src/ragnarok/training/hard_examples.py`
- Create: `tests/training/test_hard_examples.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `select_hard_examples(records, *, conf_threshold) -> list` — `records` is a list of `(item_id, max_confidence | None)` where `max_confidence` is the highest detection confidence on that frame (`None` or empty → a "missed"/no-detection frame). Returns the `item_id`s that are hard: `max_confidence is None` OR `max_confidence < conf_threshold`, preserving input order. This is the set the Roboflow client (Plan 6B) pushes back for the next dataset version (spec §12 step 6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_hard_examples.py
"""Tests for pure hard-example selection."""
from __future__ import annotations
from ragnarok.training.hard_examples import select_hard_examples


def test_selects_low_confidence_and_missed():
    records = [("a", 0.95), ("b", 0.30), ("c", None), ("d", 0.49)]
    assert select_hard_examples(records, conf_threshold=0.5) == ["b", "c", "d"]


def test_empty_when_all_confident():
    records = [("a", 0.9), ("b", 0.8)]
    assert select_hard_examples(records, conf_threshold=0.5) == []


def test_preserves_input_order():
    records = [("z", None), ("y", 0.1), ("x", 0.99)]
    assert select_hard_examples(records, conf_threshold=0.5) == ["z", "y"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_hard_examples.py -v`
Expected: FAIL — `No module named 'ragnarok.training.hard_examples'`.

- [ ] **Step 3: Implement hard_examples.py**

```python
# src/ragnarok/training/hard_examples.py
"""Hard-example mining selection (spec §12 step 6).

Pure policy: pick the frames the detector did worst on (missed entirely, or
low max confidence) to push back to Roboflow for the next dataset version. The
actual push is the Roboflow client's job (Plan 6B).
"""
from __future__ import annotations


def select_hard_examples(records, *, conf_threshold: float) -> list:
    out = []
    for item_id, max_conf in records:
        if max_conf is None or max_conf < conf_threshold:
            out.append(item_id)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_hard_examples.py -v`
Expected: PASS

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 6A).

```bash
git add src/ragnarok/training/hard_examples.py tests/training/test_hard_examples.py
git commit -m "feat(training): pure hard-example selection policy"
```

---

## Phase 6A completion checklist

- [ ] `TrainingConfig` (paths + sampling thresholds; API key via env, not config) nested in AppConfig (T1).
- [ ] Pure metrics: IoU, VOC AP@0.75 (== single-class mAP), center-error (T2).
- [ ] Benchmark harness over a labeled set with latency p50/p99 (T3).
- [ ] Smart-sampling decision (uncertainty + scene-change) (T4) + rate-limited FrameGrabber with injected writer (T5).
- [ ] Hard-example selection policy (T6).
- [ ] Full suite green; everything CI-safe (no GPU/disk/network); Scope-Boundary deferrals (6B Roboflow client, 6C ONNX/TRT export, real train, live disk capture) documented.

After merge: update memory (Phase 6A done — collection + measurement tooling ready). **Box-only follow-up to actually use it:** wire `FrameGrabber` into the worker (writer = real disk writer under `training.frames_dir`) to collect during play, label on Roboflow, then **Plan 6B** (Roboflow client: upload/export/download + push hard examples) and **Plan 6C** (ONNX/TensorRT export + engine detector backend + FP16-vs-INT8 benchmark using this harness). Natural sequence: 6B → 6C → Phase 7 (Arduino) / Phase 8 (Cyberpunk GUI).
