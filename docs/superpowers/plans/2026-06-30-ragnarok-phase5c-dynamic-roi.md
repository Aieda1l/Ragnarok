# Ragnarok Phase 5C — Dynamic-ROI (SEARCH/TRACK) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CI-safe core of the single-engine dynamic-ROI detector strategy (spec §5.2): a SEARCH/TRACK state machine + the ROI coordinate transforms (wide-letterbox in SEARCH, tight-crop-upscale in TRACK) + a planner that picks the per-frame region and maps detections back to full-frame coords — so detection puts maximum pixels-on-target without changing the fixed 384 engine.

**Architecture:** Pure, dependency-injected units in a new `detection/roi.py`: the letterbox transform (fit a wide ROI into the square engine input, preserving aspect) and the crop transform (a square crop around the predicted target, uniformly upscaled to the engine input) are pure coordinate math with exact inverse mappings; a small `RoiState` FSM persists SEARCH/TRACK and a periodic wide-rescan override; a `DynamicRoiPlanner` ties them together — given the frame size, the predicted target center, and the frame index it returns the region to capture/feed and maps the engine-space detections back. The actual cropping/upscaling of pixels and the real detector run stay in the (box-only) worker integration; this plan ships only the math + FSM + planner, all unit-tested with synthetic boxes.

**Tech Stack:** Python 3.11+, pure Python/numpy-free math (floats/ints), pydantic config. No new dependencies. The pixel crop/resize + real detector wiring is box-only (the worker uses the planner to decide the region; OpenCV/torch do the actual resize).

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is a detection-throughput/accuracy feature (spec §5.2, §18 dynamic-ROI situational-awareness risk).
- **Single static 384 engine** is the default (spec §5.2): dynamic-ROI changes only *what region* is fed to that one fixed-size engine, never the engine. The two-engine "max-FPS" mode is out of scope.
- **CI-safe always:** no GPU/display/capture/detector in unit tests. This plan is pure coordinate math + an FSM + a planner over plain numbers; the pixel crop/resize and the real detector are box-only (the worker integration is a documented deferral).
- **Coordinate correctness is the whole point:** every forward transform (full-ROI → engine input) must have an exact tested inverse (engine-space box → full-frame box), so detections land in the right place. Off-by-one / pad / scale errors here silently mis-aim — they are the primary risk.
- **Off by default:** `DynamicRoiConfig.enabled = False` → the worker keeps today's centered-ROI behavior; existing tests stay green.
- **Periodic wide rescan** (spec §5.2/§18): while TRACKing, force a SEARCH frame every N frames so incoming enemies aren't missed.
- **Frozen pydantic config**, backward-compatible TOML round-trip.
- **TDD, frequent commits, exact file paths.** Match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, frozen dataclasses, module docstrings).

## Scope Boundary (explicit deferrals)

- **Worker integration** — the actual per-frame crop+resize of captured pixels, feeding the engine, and applying the planner's map-back before tracking → box-only (needs real capture + detector + cv2 resize). This plan ships the planner the worker will call; wiring it into `worker/loop.py` is a documented follow-up.
- **Two-engine max-FPS path** (384 search + 256 track) → out of scope (spec marks it off-by-default, sub-ms gain).
- **CUDA-Graph capture / on-GPU preprocessing** → box-only optimization, unrelated to the ROI decision logic.
- **The tracker predicting the crop center "one frame ahead"** → this plan accepts the predicted center as an input (the caller can pass the IMM `lead` point); the prediction itself already exists in `aim/imm.py`. Choosing/wiring the predictor is part of the box-only worker integration.

---

## File Structure

**New files:**
- `src/ragnarok/detection/roi.py` — `letterbox_params`, `map_back_letterbox`, `crop_region_for`, `map_back_crop`, `RoiMode`, `RoiState`, `RoiPlan`, `DynamicRoiPlanner`.
- `tests/detection/test_roi_transforms.py`, `tests/detection/test_roi_fsm.py`, `tests/detection/test_roi_planner.py`

**Modified files:**
- `src/ragnarok/config/schema.py` — add `DynamicRoiConfig`, nest in `AppConfig`.
- `tests/config/test_dynamic_roi_config.py` (new).

---

## Task 1: Letterbox transform (SEARCH)

**Files:**
- Create: `src/ragnarok/detection/roi.py`
- Create: `tests/detection/test_roi_transforms.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `letterbox_params(src_w: int, src_h: int, dst: int) -> tuple[float, float, float]` — returns `(scale, pad_x, pad_y)` to fit a `src_w×src_h` region into a `dst×dst` square preserving aspect: `scale = min(dst/src_w, dst/src_h)`, the scaled image is centered with `pad_x = (dst - src_w*scale)/2`, `pad_y = (dst - src_h*scale)/2`.
  - `map_back_letterbox(box, scale, pad_x, pad_y) -> tuple[float,float,float,float]` — inverse-maps an engine-space `(x1,y1,x2,y2)` back to the **source-ROI** frame: `(c - pad)/scale` per coordinate. (The caller adds the source ROI's global offset.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_roi_transforms.py
"""Tests for dynamic-ROI coordinate transforms (pure)."""
from __future__ import annotations
from ragnarok.detection.roi import letterbox_params, map_back_letterbox


def test_letterbox_square_source_is_uniform():
    # 384x384 source into 384 -> scale 1, no pad
    assert letterbox_params(384, 384, 384) == (1.0, 0.0, 0.0)


def test_letterbox_wide_source_pads_vertically():
    # 768x384 into 384 -> scale 0.5, scaled height 192, pad_y 96, pad_x 0
    scale, px, py = letterbox_params(768, 384, 384)
    assert scale == 0.5 and px == 0.0 and py == 96.0


def test_letterbox_roundtrip_inverse():
    # A box known in source space -> forward (scale+pad) -> map_back recovers it.
    scale, px, py = letterbox_params(768, 384, 384)        # scale 0.5, py 96
    src_box = (100.0, 50.0, 200.0, 150.0)
    fwd = (src_box[0] * scale + px, src_box[1] * scale + py,
           src_box[2] * scale + px, src_box[3] * scale + py)
    back = map_back_letterbox(fwd, scale, px, py)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, src_box))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_roi_transforms.py -v`
Expected: FAIL — `No module named 'ragnarok.detection.roi'`.

- [ ] **Step 3: Implement the letterbox transform**

```python
# src/ragnarok/detection/roi.py
"""Dynamic-ROI (SEARCH/TRACK) coordinate math + FSM + planner (spec §5.2).

Pure: decides what region of the captured frame to feed the fixed 384 engine
(wide letterbox in SEARCH; tight square crop upscaled in TRACK) and maps the
engine-space detections back to full-frame coordinates. The actual pixel
crop/resize + the real detector are box-only (worker integration).
"""
from __future__ import annotations


def letterbox_params(src_w: int, src_h: int, dst: int) -> tuple[float, float, float]:
    scale = min(dst / src_w, dst / src_h)
    pad_x = (dst - src_w * scale) / 2.0
    pad_y = (dst - src_h * scale) / 2.0
    return (scale, pad_x, pad_y)


def map_back_letterbox(box, scale: float, pad_x: float, pad_y: float):
    x1, y1, x2, y2 = box
    return ((x1 - pad_x) / scale, (y1 - pad_y) / scale,
            (x2 - pad_x) / scale, (y2 - pad_y) / scale)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/detection/test_roi_transforms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/detection/roi.py tests/detection/test_roi_transforms.py
git commit -m "feat(detection): dynamic-ROI letterbox transform (SEARCH)"
```

---

## Task 2: Crop transform (TRACK)

**Files:**
- Modify: `src/ragnarok/detection/roi.py`
- Test: `tests/detection/test_roi_transforms.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `crop_region_for(center, size: int, frame_w: int, frame_h: int) -> tuple[int,int,int,int]` — a `size×size` square crop centered on `center=(cx,cy)`, clamped so it stays inside `[0,frame_w]×[0,frame_h]`; returns `(x0, y0, size, size)`. Assumes `size <= min(frame_w, frame_h)` (the caller ensures it; the function clamps the origin regardless).
  - `map_back_crop(box, crop_region, dst: int) -> tuple[float,float,float,float]` — inverse of "crop a `size×size` region and uniformly upscale to `dst×dst`": with `(x0,y0,size,_)=crop_region` and `r = size/dst`, returns `(bx*r + x0, by*r + y0, ...)` for an engine-space `(x1,y1,x2,y2)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/detection/test_roi_transforms.py
from ragnarok.detection.roi import crop_region_for, map_back_crop


def test_crop_centered_when_room():
    assert crop_region_for((500, 400), 192, 1920, 1080) == (404, 304, 192, 192)


def test_crop_clamped_at_top_left():
    assert crop_region_for((10, 10), 192, 1920, 1080) == (0, 0, 192, 192)


def test_crop_clamped_at_bottom_right():
    assert crop_region_for((1915, 1075), 192, 1920, 1080) == (1728, 888, 192, 192)


def test_crop_map_back_upscales_and_offsets():
    crop = (404, 304, 192, 192)                 # size 192 fed to a 384 engine -> r=0.5
    # an engine-space box at (0,0,384,384) maps to the full crop region
    back = map_back_crop((0.0, 0.0, 384.0, 384.0), crop, 384)
    assert back == (404.0, 304.0, 404.0 + 192.0, 304.0 + 192.0)


def test_crop_map_back_roundtrip():
    crop = (404, 304, 192, 192)
    full_box = (450.0, 350.0, 500.0, 420.0)     # known full-frame box inside the crop
    r = 192 / 384
    eng = ((full_box[0] - 404) / r, (full_box[1] - 304) / r,
           (full_box[2] - 404) / r, (full_box[3] - 304) / r)
    back = map_back_crop(eng, crop, 384)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, full_box))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_roi_transforms.py -k crop -v`
Expected: FAIL — `cannot import name 'crop_region_for'`.

- [ ] **Step 3: Implement the crop transform**

Append to `src/ragnarok/detection/roi.py`:

```python
def crop_region_for(center, size: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    cx, cy = center
    x0 = int(round(cx - size / 2.0))
    y0 = int(round(cy - size / 2.0))
    x0 = max(0, min(x0, frame_w - size))
    y0 = max(0, min(y0, frame_h - size))
    return (x0, y0, size, size)


def map_back_crop(box, crop_region, dst: int):
    x0, y0, size, _ = crop_region
    r = size / dst
    x1, y1, x2, y2 = box
    return (x1 * r + x0, y1 * r + y0, x2 * r + x0, y2 * r + y0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/detection/test_roi_transforms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/detection/roi.py tests/detection/test_roi_transforms.py
git commit -m "feat(detection): dynamic-ROI crop transform (TRACK)"
```

---

## Task 3: SEARCH/TRACK FSM + rescan

**Files:**
- Modify: `src/ragnarok/detection/roi.py`
- Test: `tests/detection/test_roi_fsm.py`

**Interfaces:**
- Consumes: nothing (stdlib `enum`).
- Produces:
  - `RoiMode(str, Enum)`: `SEARCH = "search"`, `TRACK = "track"`.
  - `RoiState(*, max_missed: int, rescan_interval: int)` with:
    - `.mode` property (starts `SEARCH`).
    - `update(*, has_lock: bool) -> RoiMode` — SEARCH→TRACK on `has_lock`; in TRACK, a frame without a lock increments a missed counter and reverts to SEARCH at `max_missed` (counter resets on lock or on reverting). Returns the new mode.
    - `wants_rescan(frame_index: int) -> bool` — `True` only while in TRACK when `rescan_interval > 0 and frame_index % rescan_interval == 0` (a transient wide-rescan that does NOT change `mode`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/detection/test_roi_fsm.py
"""Tests for the dynamic-ROI SEARCH/TRACK FSM."""
from __future__ import annotations
from ragnarok.detection.roi import RoiMode, RoiState


def test_starts_in_search():
    assert RoiState(max_missed=3, rescan_interval=30).mode == RoiMode.SEARCH


def test_lock_enters_track():
    s = RoiState(max_missed=3, rescan_interval=30)
    assert s.update(has_lock=True) == RoiMode.TRACK
    assert s.mode == RoiMode.TRACK


def test_reverts_to_search_after_max_missed():
    s = RoiState(max_missed=2, rescan_interval=0)
    s.update(has_lock=True)               # TRACK
    assert s.update(has_lock=False) == RoiMode.TRACK   # 1 missed, still tracking
    assert s.update(has_lock=False) == RoiMode.SEARCH  # 2 missed -> SEARCH


def test_missed_counter_resets_on_relock():
    s = RoiState(max_missed=2, rescan_interval=0)
    s.update(has_lock=True)
    s.update(has_lock=False)              # 1 missed
    s.update(has_lock=True)               # relock -> reset
    assert s.update(has_lock=False) == RoiMode.TRACK   # only 1 missed again


def test_rescan_only_in_track_on_interval():
    s = RoiState(max_missed=3, rescan_interval=10)
    assert s.wants_rescan(10) is False    # still SEARCH -> no rescan concept
    s.update(has_lock=True)               # TRACK
    assert s.wants_rescan(10) is True     # 10 % 10 == 0
    assert s.wants_rescan(13) is False
    assert s.mode == RoiMode.TRACK        # rescan query does not change mode


def test_rescan_disabled_when_interval_zero():
    s = RoiState(max_missed=3, rescan_interval=0)
    s.update(has_lock=True)
    assert s.wants_rescan(0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/detection/test_roi_fsm.py -v`
Expected: FAIL — `cannot import name 'RoiMode'`.

- [ ] **Step 3: Implement the FSM**

Append to `src/ragnarok/detection/roi.py`:

```python
from enum import Enum


class RoiMode(str, Enum):
    SEARCH = "search"
    TRACK = "track"


class RoiState:
    def __init__(self, *, max_missed: int, rescan_interval: int) -> None:
        self._max_missed = max_missed
        self._rescan = rescan_interval
        self._mode = RoiMode.SEARCH
        self._missed = 0

    @property
    def mode(self) -> RoiMode:
        return self._mode

    def update(self, *, has_lock: bool) -> RoiMode:
        if self._mode == RoiMode.SEARCH:
            if has_lock:
                self._mode = RoiMode.TRACK
                self._missed = 0
        else:  # TRACK
            if has_lock:
                self._missed = 0
            else:
                self._missed += 1
                if self._missed >= self._max_missed:
                    self._mode = RoiMode.SEARCH
                    self._missed = 0
        return self._mode

    def wants_rescan(self, frame_index: int) -> bool:
        return (self._mode == RoiMode.TRACK and self._rescan > 0
                and frame_index % self._rescan == 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/detection/test_roi_fsm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/detection/roi.py tests/detection/test_roi_fsm.py
git commit -m "feat(detection): dynamic-ROI SEARCH/TRACK FSM with periodic rescan"
```

---

## Task 4: DynamicRoiConfig + DynamicRoiPlanner

**Files:**
- Modify: `src/ragnarok/config/schema.py`, `src/ragnarok/detection/roi.py`
- Test: `tests/config/test_dynamic_roi_config.py`, `tests/detection/test_roi_planner.py`

**Interfaces:**
- Consumes: `letterbox_params`/`map_back_letterbox`/`crop_region_for`/`map_back_crop`/`RoiMode`/`RoiState` (T1–T3).
- Produces:
  - `DynamicRoiConfig` (frozen): `enabled: bool = False`, `track_roi_size: int = Field(default=192, ge=32)`, `model_input_px: int = Field(default=384, ge=64)`, `max_missed_frames: int = Field(default=5, ge=1)`, `rescan_interval_frames: int = Field(default=30, ge=0)` (0 = no rescan). Nested as `AppConfig.dynamic_roi`.
  - `@dataclass(frozen=True) RoiPlan`: `mode: RoiMode`, `region: tuple[int,int,int,int]` (x0,y0,w,h in full-frame px to capture/feed), `letterboxed: bool` (True for SEARCH, False for TRACK crop).
  - `DynamicRoiPlanner(cfg)` with:
    - `plan(*, frame_w, frame_h, target_center, frame_index, has_lock) -> RoiPlan` — advances the `RoiState` with `has_lock`; if the effective mode is SEARCH (state SEARCH **or** a due rescan) → region is the full frame, `letterboxed=True`; if TRACK → a `crop_region_for(target_center, track_roi_size, ...)`, `letterboxed=False`.
    - `map_back(box, plan) -> tuple[float,float,float,float]` — maps an engine-space box to full-frame coords using the letterbox params (for a SEARCH/full region, origin 0,0) or the crop transform (for TRACK), per `plan.letterboxed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_dynamic_roi_config.py
"""Tests for DynamicRoiConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DynamicRoiConfig, AppConfig


def test_defaults_off():
    d = DynamicRoiConfig()
    assert d.enabled is False
    assert d.track_roi_size == 192 and d.model_input_px == 384
    assert d.max_missed_frames == 5 and d.rescan_interval_frames == 30


def test_bounds():
    with pytest.raises(ValidationError):
        DynamicRoiConfig(max_missed_frames=0)
    with pytest.raises(ValidationError):
        DynamicRoiConfig(track_roi_size=16)


def test_nested_backward_compatible():
    assert isinstance(AppConfig().dynamic_roi, DynamicRoiConfig)
    assert AppConfig(detection={"model": "nano"}).dynamic_roi.enabled is False
```

```python
# tests/detection/test_roi_planner.py
"""Tests for the DynamicRoiPlanner (FSM + transforms tied together)."""
from __future__ import annotations
from ragnarok.config.schema import DynamicRoiConfig
from ragnarok.detection.roi import DynamicRoiPlanner, RoiMode


def test_search_plan_is_full_frame_letterboxed():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384))
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=None,
                  frame_index=0, has_lock=False)
    assert plan.mode == RoiMode.SEARCH
    assert plan.region == (0, 0, 1920, 1080) and plan.letterboxed is True


def test_track_plan_is_crop_after_lock():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384,
                                           rescan_interval_frames=0))
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=0, has_lock=True)                    # -> TRACK
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                  frame_index=1, has_lock=True)
    assert plan.mode == RoiMode.TRACK
    assert plan.region == (404, 304, 192, 192) and plan.letterboxed is False


def test_rescan_forces_full_frame_while_tracking():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, rescan_interval_frames=5))
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=1, has_lock=True)                    # TRACK
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                  frame_index=5, has_lock=True)             # 5 % 5 == 0 -> rescan
    assert plan.mode == RoiMode.TRACK                       # logical mode unchanged
    assert plan.region == (0, 0, 1920, 1080) and plan.letterboxed is True


def test_map_back_search_then_track():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384,
                                           rescan_interval_frames=0))
    # SEARCH: full 1920x1080 letterboxed into 384
    sp = p.plan(frame_w=1920, frame_h=1080, target_center=None,
                frame_index=0, has_lock=False)
    full = p.map_back((192.0, 0.0, 384.0, 216.0), sp)       # engine-space box
    # 1920x1080 -> scale 0.2, pad_y (384-216)/2=84; map back of (192,0,..) etc.
    assert full[0] > 0.0 and full[2] > full[0]
    # TRACK crop map-back lands inside the crop region
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=1, has_lock=True)
    tp = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                frame_index=2, has_lock=True)
    tback = p.map_back((0.0, 0.0, 384.0, 384.0), tp)
    assert tback == (404.0, 304.0, 404.0 + 192.0, 304.0 + 192.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_dynamic_roi_config.py tests/detection/test_roi_planner.py -v`
Expected: FAIL — `cannot import name 'DynamicRoiConfig'` / `DynamicRoiPlanner`.

- [ ] **Step 3: Implement config, RoiPlan, and the planner**

In `src/ragnarok/config/schema.py`, add (before `AppConfig`):

```python
class DynamicRoiConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    track_roi_size: int = Field(default=192, ge=32)
    model_input_px: int = Field(default=384, ge=64)
    max_missed_frames: int = Field(default=5, ge=1)
    rescan_interval_frames: int = Field(default=30, ge=0)   # 0 = no periodic rescan
```

Add to `AppConfig` (after `arduino`):

```python
    dynamic_roi: DynamicRoiConfig = DynamicRoiConfig()
```

Append to `src/ragnarok/detection/roi.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RoiPlan:
    mode: RoiMode
    region: tuple[int, int, int, int]     # (x0, y0, w, h) in full-frame pixels
    letterboxed: bool                     # True = SEARCH (letterbox), False = TRACK (crop)


class DynamicRoiPlanner:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._dst = cfg.model_input_px
        self._state = RoiState(max_missed=cfg.max_missed_frames,
                               rescan_interval=cfg.rescan_interval_frames)

    def plan(self, *, frame_w: int, frame_h: int, target_center,
             frame_index: int, has_lock: bool) -> RoiPlan:
        self._state.update(has_lock=has_lock)
        if self._state.mode == RoiMode.SEARCH or self._state.wants_rescan(frame_index):
            return RoiPlan(mode=self._state.mode, region=(0, 0, frame_w, frame_h),
                           letterboxed=True)
        region = crop_region_for(target_center, self._cfg.track_roi_size, frame_w, frame_h)
        return RoiPlan(mode=self._state.mode, region=region, letterboxed=False)

    def map_back(self, box, plan: RoiPlan):
        if plan.letterboxed:
            _x0, _y0, w, h = plan.region
            scale, pad_x, pad_y = letterbox_params(w, h, self._dst)
            mx1, my1, mx2, my2 = map_back_letterbox(box, scale, pad_x, pad_y)
            return (mx1 + plan.region[0], my1 + plan.region[1],
                    mx2 + plan.region[0], my2 + plan.region[1])
        return map_back_crop(box, plan.region, self._dst)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config tests/detection/test_roi_planner.py -q`
Expected: PASS (incl. TOML round-trip).

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 5C).

```bash
git add src/ragnarok/config/schema.py src/ragnarok/detection/roi.py tests/config/test_dynamic_roi_config.py tests/detection/test_roi_planner.py
git commit -m "feat(detection): DynamicRoiConfig + DynamicRoiPlanner (FSM + transforms + map-back)"
```

---

## Phase 5C completion checklist

- [ ] Letterbox transform + inverse (SEARCH) (T1); crop transform + inverse (TRACK) (T2) — exact round-trips tested.
- [ ] SEARCH/TRACK FSM with miss-counter revert + periodic rescan override (T3).
- [ ] `DynamicRoiConfig` (off by default) + `DynamicRoiPlanner` tying FSM + transforms + map-back (T4).
- [ ] Full suite green; CI-safe (pure math/FSM, no GPU/capture/detector); Scope-Boundary deferrals (worker pixel crop/resize integration, two-engine path, CUDA-graph, predictor wiring) documented.

After merge: update memory (Phase 5C done — dynamic-ROI decision core ready; off by default). **Box-only worker integration:** in `worker/loop.py`, when `dynamic_roi.enabled`, call `planner.plan(...)` with the IMM-predicted target center (`aim/imm.py` lead), crop/resize the captured frame to the planned region (cv2), run the 384 engine, then `planner.map_back(...)` each detection to full-frame coords before tracking; validate pixels-on-target/accuracy gains live. Natural next: **Phase 8 (Cyberpunk GUI)** — the capstone (consumes 5A diagnostics + apply_seeds, 5B calibration, 6/7 backends, this ROI mode); it needs the user's visual/scoping input, so scope it interactively. Phase 7B (firmware + remaining transports) is the other box-only track.
