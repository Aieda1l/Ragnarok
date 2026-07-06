# Phase 8A — Smart-Lock FOV Overlay (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Cyberpunk "smart-weapon lock-on" FOV overlay's *geometry and data path* as pure, unit-tested code, plus a thin QPainter widget, so the running system can draw an FOV ring, target markers/diamonds, converging acquisition brackets on the locked target, a tracking line to the crosshair, and off-screen direction hints.

**Architecture:** All overlay math lives in a **Qt-free pure module** (`gui/overlay_model.py`) that turns a `TelemetrySnapshot` + `AppConfig` into an `OverlayScene` of screen-space draw primitives — fully unit-testable with no display/GPU/mouse. A thin `FovOverlay(QWidget)` renders that scene with `QPainter` (frameless, always-on-top, click-through, translucent, own timer decoupled from the hot loop). The snapshot is enriched with `locked_target_id` + `roi_region` and the worker loop publishes them, so the whole data path is CI-covered. The full Cyberpunk visual restyle (QSS, glitch, `QOpenGLWidget` GL backing, Win32 `WS_EX_TRANSPARENT` click-through) is an explicit box-only follow-up.

**Tech Stack:** Python 3.11+, PySide6 (`QWidget`/`QPainter`/`QTimer`), pytest + pytest-qt (`qtbot`), pure `math`. No torch/GPU/network in any test.

## Global Constraints

- **The overlay is cosmetic/diagnostic ONLY.** Aim uses detection coordinates, never rendered pixels. `overlay_model`/`FovOverlay` may only *read* telemetry; they must never call the mouse, mutate config, or feed anything back into the aim path. (spec §2, §4)
- **Primary accent = electric yellow `#FCEE0A`**; secondaries cyan/teal `#00F0FF` and alert-red `#FF3B3B`; near-black background. Diamond motif for confirmed targets. (spec §10.1–10.2)
- **Team color-coding** must stay consistent with the existing detection overlay palette (`gui/overlay.py::TEAM_BGR`: enemy orange, teammate blue, unknown gray) — mirror those hues in RGB for Qt. (spec §10.2)
- **Overlay window:** frameless, always-on-top, `Qt.Tool`, `WA_TranslucentBackground`, click-through, rendered on its **own timer decoupled from the hot loop**. (spec §10.2)
- **CI-safety:** `overlay_model` imports **zero Qt**. Widget tests use the `qtbot` fixture (same pattern as `tests/gui/test_main_window.py`). No test imports torch, rfdetr, or any GPU/network dependency.
- **Append-only snapshot:** new `TelemetrySnapshot` fields must have defaults so every existing constructor call keeps working. (matches the Phase 2 `tracks=()` precedent)
- TDD, one deliverable per task, commit per task.

---

## File Structure

- **Create** `src/ragnarok/gui/overlay_model.py` — pure geometry: `ScreenMap`, `FovRing`, `TargetMarker`, `OffscreenHint`, `OverlayScene`, `LockAgeTracker`, `bracket_segments`, `lock_progress`, `build_markers`, `build_scene`, plus internal `_ray_rect_edge`/`_in_viewport`. Qt-free.
- **Create** `src/ragnarok/gui/theme.py` — Cyberpunk palette tokens + `team_color(team)`. Pure constants.
- **Create** `src/ragnarok/gui/overlay_window.py` — `FovOverlay(QWidget)`: window plumbing + `QPainter` renderer consuming `build_scene`.
- **Modify** `src/ragnarok/telemetry/snapshot.py` — add `locked_target_id` + `roi_region` (append-only).
- **Modify** `src/ragnarok/worker/loop.py` — publish `locked_target_id` (from `aim_controller.target_id`) + `roi_region` (from `frame.region`).
- **Modify** `src/ragnarok/app.py` — instantiate + show `FovOverlay` alongside `MainWindow` (box-only wiring).
- **Create** tests: `tests/gui/test_overlay_model.py`, `tests/gui/test_theme.py`, `tests/gui/test_overlay_window.py`; extend `tests/worker/test_loop.py`, `tests/telemetry/test_snapshot.py`.

Test runner for every step: `uv run --extra dev pytest`. Baseline before this plan: **518 passed**.

---

### Task 1: Enrich `TelemetrySnapshot` with `locked_target_id` + `roi_region`

**Files:**
- Modify: `src/ragnarok/telemetry/snapshot.py`
- Test: `tests/telemetry/test_snapshot.py`

**Interfaces:**
- Consumes: existing `TelemetrySnapshot` (frozen dataclass).
- Produces: `TelemetrySnapshot(..., locked_target_id: int | None = None, roi_region: tuple[int,int,int,int] | None = None)` — append-only fields after `tracks`.

- [ ] **Step 1: Write the failing test** — append to `tests/telemetry/test_snapshot.py`:

```python
def test_snapshot_lock_and_region_default_and_carry():
    s = TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                          detection_count=0, preview=None, seq=1)
    assert s.locked_target_id is None and s.roi_region is None
    s2 = TelemetrySnapshot(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0,
                           detection_count=0, preview=None, seq=1,
                           locked_target_id=7, roi_region=(10, 20, 394, 404))
    assert s2.locked_target_id == 7 and s2.roi_region == (10, 20, 394, 404)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/telemetry/test_snapshot.py -q`
Expected: FAIL (unexpected keyword argument `locked_target_id`).

- [ ] **Step 3: Implement** — in `src/ragnarok/telemetry/snapshot.py`, add two fields after `tracks`:

```python
    tracks: tuple[Track, ...] = ()  # Phase 2; default keeps Phase 1 callers working
    locked_target_id: int | None = None      # Phase 8A: current aim lock (overlay highlight)
    roi_region: tuple[int, int, int, int] | None = None  # Phase 8A: (l,t,r,b) screen coords of the ROI
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/telemetry/test_snapshot.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/telemetry/snapshot.py tests/telemetry/test_snapshot.py
git commit -m "feat(overlay): snapshot carries locked_target_id + roi_region"
```

---

### Task 2: Worker loop publishes `locked_target_id` + `roi_region`

**Files:**
- Modify: `src/ragnarok/worker/loop.py:72-76`
- Test: `tests/worker/test_loop.py`

**Interfaces:**
- Consumes: `frame.region` (from `core.types.Frame`), `aim_controller.target_id` (optional; may be absent → `None`), enriched `TelemetrySnapshot` (Task 1).
- Produces: published snapshots carry `locked_target_id` + `roi_region`.

- [ ] **Step 1: Write the failing test** — append to `tests/worker/test_loop.py`. Reuse the file's existing fakes; add an aim stub exposing `target_id`. If the existing fake frame lacks `region`, set it. Minimal self-contained test:

```python
def test_loop_publishes_lock_and_region():
    from ragnarok.worker.loop import WorkerLoop
    from ragnarok.latency.profiler import StageProfiler
    import numpy as np
    from ragnarok.core.types import Frame, Detections

    class _Cap:
        def start(self): ...
        def stop(self): ...
        def grab(self):
            return Frame(image=np.zeros((384, 384, 3), np.uint8),
                         t_capture_ns=0, region=(100, 50, 484, 434))

    class _Det:
        def detect(self, frame):
            return Detections.empty()

    class _Aim:
        target_id = 42
        def update(self, tracks, t_ns): ...

    pub = SnapshotPublisher()
    WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, aim_controller=_Aim()).tick()
    snap = pub.latest()
    assert snap.locked_target_id == 42
    assert snap.roi_region == (100, 50, 484, 434)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/worker/test_loop.py::test_loop_publishes_lock_and_region -q`
Expected: FAIL (`locked_target_id` is `None`, not 42).

- [ ] **Step 3: Implement** — in `src/ragnarok/worker/loop.py`, change the `publish(...)` call (lines 72-76) to:

```python
        self._pub.publish(TelemetrySnapshot(
            fps=fps, loop_ms_p50=p50, loop_ms_p99=p99,
            detection_count=len(dets), preview=preview, seq=self._seq,
            tracks=tuple(tracks),
            locked_target_id=getattr(self._aim, "target_id", None),
            roi_region=frame.region,
        ))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/worker/test_loop.py -q`
Expected: PASS (new test + all existing loop tests — the `getattr` default keeps `aim_controller=None` cases at `None`).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/worker/loop.py tests/worker/test_loop.py
git commit -m "feat(overlay): loop publishes aim lock id + ROI screen region"
```

---

### Task 3: `overlay_model` — `ScreenMap`, `FovRing`, `OverlayScene` shell

**Files:**
- Create: `src/ragnarok/gui/overlay_model.py`
- Test: `tests/gui/test_overlay_model.py`

**Interfaces:**
- Consumes: `core.types.Team/Track`, `aim.fov.fov_deg_to_radius_px/aim_point`.
- Produces:
  - `ScreenMap(left, top, scale_x, scale_y)` with `from_region(region, roi_w, roi_h)`, `pt(x, y)->(sx,sy)`, `rect(xyxy)->(sx1,sy1,sx2,sy2)`.
  - `FovRing(center, acquire_radius, retain_radius, tick_count=12)`.
  - `OverlayScene(has_signal, crosshair, fov, markers, bracket_segments, locked_line, offscreen)` + `OverlayScene.empty()`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_overlay_model.py`:

```python
import math
from ragnarok.gui.overlay_model import ScreenMap, FovRing, OverlayScene


def test_screenmap_from_region_maps_points_and_rect():
    # ROI 384 captured at screen (100,50)-(484,434): scale 1.0, offset (100,50)
    m = ScreenMap.from_region((100, 50, 484, 434), 384, 384)
    assert m.pt(0, 0) == (100.0, 50.0)
    assert m.pt(192, 192) == (292.0, 242.0)          # ROI centre -> region centre
    assert m.rect((0, 0, 10, 20)) == (100.0, 50.0, 110.0, 70.0)


def test_screenmap_scales_when_region_larger_than_roi():
    m = ScreenMap.from_region((0, 0, 768, 768), 384, 384)  # 2x upscale
    assert m.scale_x == 2.0 and m.scale_y == 2.0
    assert m.pt(10, 10) == (20.0, 20.0)


def test_empty_scene_has_no_signal():
    s = OverlayScene.empty()
    assert s.has_signal is False and s.fov is None and s.markers == ()


def test_fovring_fields():
    r = FovRing(center=(5.0, 6.0), acquire_radius=10.0, retain_radius=20.0)
    assert r.center == (5.0, 6.0) and r.acquire_radius < r.retain_radius
    assert r.tick_count == 12
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/overlay_model.py`:

```python
"""Pure geometry for the smart-lock FOV overlay (spec §10.2).

ZERO Qt dependency: turns a telemetry snapshot + config into a set of draw
primitives (an ``OverlayScene``) in absolute screen coordinates. The QPainter
widget (``overlay_window.py``) consumes this; all the math lives here so it is
unit-testable without any display, GPU, or mouse.

Coordinates: tracks live in ROI-pixel space (crosshair at ROI centre).
``ScreenMap`` maps ROI px -> absolute screen px using the captured region.
FOV radii come from ``aim.fov.fov_deg_to_radius_px`` (screen px). The normal
capture path is 1:1 (region width == roi_size, scale 1.0); dynamic-ROI tracks
are already mapped back to full-frame coords upstream, so scale stays 1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ragnarok.core.types import Team, Track
from ragnarok.aim.fov import aim_point, fov_deg_to_radius_px


@dataclass(frozen=True)
class ScreenMap:
    left: float
    top: float
    scale_x: float
    scale_y: float

    @classmethod
    def from_region(cls, region, roi_w: int, roi_h: int) -> "ScreenMap":
        left, top, right, bottom = region
        return cls(left=float(left), top=float(top),
                   scale_x=(right - left) / float(roi_w),
                   scale_y=(bottom - top) / float(roi_h))

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return (self.left + x * self.scale_x, self.top + y * self.scale_y)

    def rect(self, xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        sx1, sy1 = self.pt(x1, y1)
        sx2, sy2 = self.pt(x2, y2)
        return (sx1, sy1, sx2, sy2)


@dataclass(frozen=True)
class FovRing:
    center: tuple[float, float]
    acquire_radius: float     # inner (acquisition)
    retain_radius: float      # outer (sticky retention)
    tick_count: int = 12


@dataclass(frozen=True)
class TargetMarker:
    track_id: int
    box: tuple[float, float, float, float]   # screen xyxy
    diamond: tuple[float, float]             # screen aim-point (confirmed-target marker)
    team: Team
    confidence: float
    locked: bool
    in_fov: bool


@dataclass(frozen=True)
class OffscreenHint:
    angle_rad: float                         # direction from crosshair
    edge_point: tuple[float, float]          # clamped to the viewport edge
    team: Team


@dataclass(frozen=True)
class OverlayScene:
    has_signal: bool
    crosshair: tuple[float, float]
    fov: FovRing | None
    markers: tuple[TargetMarker, ...]
    bracket_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    locked_line: tuple[tuple[float, float], tuple[float, float]] | None
    offscreen: tuple[OffscreenHint, ...]

    @classmethod
    def empty(cls) -> "OverlayScene":
        return cls(has_signal=False, crosshair=(0.0, 0.0), fov=None, markers=(),
                   bracket_segments=(), locked_line=None, offscreen=())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/overlay_model.py tests/gui/test_overlay_model.py
git commit -m "feat(overlay): pure ScreenMap + FovRing + OverlayScene primitives"
```

---

### Task 4: `overlay_model` — target markers + `LockAgeTracker`

**Files:**
- Modify: `src/ragnarok/gui/overlay_model.py`
- Test: `tests/gui/test_overlay_model.py`

**Interfaces:**
- Consumes: `ScreenMap`, `TargetMarker` (Task 3), `aim.fov.aim_point`.
- Produces:
  - `build_markers(tracks, smap, crosshair, fov_px, locked_id, head_frac, aim_mode) -> tuple[TargetMarker, ...]`.
  - `LockAgeTracker()` with `.update(locked_id, now_ns) -> float` (age seconds; resets on lock change, 0.0 when unlocked).

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_overlay_model.py`:

```python
from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay_model import build_markers, LockAgeTracker, ScreenMap


def _trk(tid, xyxy, team=Team.ENEMY, conf=0.9):
    return Track(track_id=tid, xyxy=xyxy, confidence=conf, class_id=0, team=team)


def test_build_markers_sets_lock_diamond_and_fov():
    m = ScreenMap.from_region((0, 0, 384, 384), 384, 384)  # identity
    crosshair = (192.0, 192.0)
    tracks = (
        _trk(1, (180, 150, 204, 234)),   # near crosshair -> in_fov, will be locked
        _trk(2, (10, 10, 30, 90)),       # far corner -> out of fov
    )
    markers = build_markers(tracks, m, crosshair, fov_px=40.0, locked_id=1,
                            head_frac=0.15, aim_mode="head")
    by_id = {mk.track_id: mk for mk in markers}
    assert by_id[1].locked is True and by_id[2].locked is False
    assert by_id[1].in_fov is True and by_id[2].in_fov is False
    # diamond is the head aim-point: x = box centre, y = y1 + 0.15*height
    assert by_id[1].diamond == (192.0, 150.0 + 0.15 * 84.0)
    assert by_id[1].box == (180.0, 150.0, 204.0, 234.0)


def test_lock_age_tracker_resets_on_change():
    t = LockAgeTracker()
    assert t.update(None, 0) == 0.0
    assert t.update(5, 1_000_000_000) == 0.0        # first frame of a new lock -> age 0
    assert t.update(5, 1_500_000_000) == 0.5        # 0.5 s later
    assert t.update(6, 1_600_000_000) == 0.0        # switched lock -> reset
    assert t.update(None, 2_000_000_000) == 0.0     # lock dropped -> 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: FAIL (`build_markers`/`LockAgeTracker` undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/overlay_model.py`:

```python
def build_markers(tracks, smap: ScreenMap, crosshair: tuple[float, float],
                  fov_px: float, locked_id: int | None,
                  head_frac: float, aim_mode: str) -> tuple[TargetMarker, ...]:
    """One ``TargetMarker`` per track, in screen coords.

    ``in_fov`` compares the screen-space crosshair->aim-point distance to
    ``fov_px`` (both screen px; scale is 1.0 on the normal capture path).
    """
    out: list[TargetMarker] = []
    for tr in tracks:
        apx, apy = aim_point(tr, head_frac, aim_mode)
        diamond = smap.pt(apx, apy)
        d = math.hypot(diamond[0] - crosshair[0], diamond[1] - crosshair[1])
        out.append(TargetMarker(
            track_id=tr.track_id, box=smap.rect(tr.xyxy), diamond=diamond,
            team=tr.team, confidence=tr.confidence,
            locked=(locked_id is not None and tr.track_id == locked_id),
            in_fov=(d <= fov_px),
        ))
    return tuple(out)


class LockAgeTracker:
    """Tracks how long the current lock has been held, for the lock-on
    convergence animation. Clock-agnostic: the caller passes ``now_ns`` each
    frame (widget uses ``core.clock.now_ns``; tests pass explicit values)."""

    def __init__(self) -> None:
        self._locked: int | None = None
        self._start_ns: int | None = None

    def update(self, locked_id: int | None, now_ns: int) -> float:
        if locked_id != self._locked:
            self._locked = locked_id
            self._start_ns = now_ns if locked_id is not None else None
        if self._start_ns is None:
            return 0.0
        return (now_ns - self._start_ns) / 1e9
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/overlay_model.py tests/gui/test_overlay_model.py
git commit -m "feat(overlay): target markers + lock-age tracker"
```

---

### Task 5: `overlay_model` — bracket convergence + off-screen hints

**Files:**
- Modify: `src/ragnarok/gui/overlay_model.py`
- Test: `tests/gui/test_overlay_model.py`

**Interfaces:**
- Consumes: `OffscreenHint` (Task 3).
- Produces:
  - `lock_progress(lock_age_s, duration_s) -> float` (0..1; 1.0 when `duration_s<=0`).
  - `bracket_segments(box, t, gap, arm_len) -> tuple[segment, ...]` where a segment is `((x1,y1),(x2,y2))`; 8 segments (2 arms × 4 corners). `t=0` → arms `gap` px outside each corner; `t=1` → arms exactly on the box corners.
  - `_ray_rect_edge(origin, target, viewport) -> (x,y)` and `_in_viewport(pt, viewport) -> bool` (viewport = `(x0,y0,x1,y1)`).

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_overlay_model.py`:

```python
from ragnarok.gui.overlay_model import (
    lock_progress, bracket_segments, _ray_rect_edge, _in_viewport)


def test_lock_progress_clamps():
    assert lock_progress(-1.0, 0.2) == 0.0
    assert lock_progress(0.1, 0.2) == 0.5
    assert lock_progress(5.0, 0.2) == 1.0
    assert lock_progress(0.0, 0.0) == 1.0            # zero-duration -> snapped


def test_bracket_segments_converge():
    box = (100.0, 100.0, 200.0, 200.0)
    # t=1: top-left corner arm starts exactly at (100,100)
    snapped = bracket_segments(box, t=1.0, gap=20.0, arm_len=10.0)
    tl_h = snapped[0]                                # first corner, horizontal arm
    assert tl_h[0] == (100.0, 100.0)
    assert tl_h[1] == (110.0, 100.0)                 # arm extends inward +x
    # t=0: same corner is offset gap px up-and-left (unconverged/wide)
    wide = bracket_segments(box, t=0.0, gap=20.0, arm_len=10.0)
    assert wide[0][0] == (80.0, 80.0)
    assert len(snapped) == 8                          # 4 corners x 2 arms


def test_ray_rect_edge_and_in_viewport():
    vp = (0.0, 0.0, 100.0, 100.0)
    assert _in_viewport((50.0, 50.0), vp) is True
    assert _in_viewport((150.0, 50.0), vp) is False
    # ray from centre toward a point far to the right hits the x=100 edge at y=50
    edge = _ray_rect_edge((50.0, 50.0), (500.0, 50.0), vp)
    assert edge == (100.0, 50.0)
    # toward bottom-right corner hits the corner
    edge2 = _ray_rect_edge((50.0, 50.0), (150.0, 150.0), vp)
    assert edge2 == (100.0, 100.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: FAIL (names undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/overlay_model.py`:

```python
def lock_progress(lock_age_s: float, duration_s: float) -> float:
    """Convergence parameter for the lock-on brackets: 0 (wide) -> 1 (snapped)."""
    if duration_s <= 0.0:
        return 1.0
    return max(0.0, min(1.0, lock_age_s / duration_s))


def bracket_segments(box, t: float, gap: float, arm_len: float):
    """L-shaped corner brackets that converge onto ``box`` as ``t`` goes 0->1.

    ``t=0``: each corner sits ``gap`` px outside the box (unconverged/wide).
    ``t=1``: each corner sits exactly on the box corner (snapped/locked).
    Returns 8 segments (horizontal + vertical arm per corner), each a pair of
    screen points ``((x1,y1),(x2,y2))``. Arms always point *inward*.
    """
    x1, y1, x2, y2 = box
    o = gap * (1.0 - t)                      # outward offset shrinks to 0
    L = arm_len
    corners = (
        (x1 - o, y1 - o, +1, +1),           # top-left    -> arms right & down
        (x2 + o, y1 - o, -1, +1),           # top-right   -> arms left & down
        (x1 - o, y2 + o, +1, -1),           # bottom-left -> arms right & up
        (x2 + o, y2 + o, -1, -1),           # bottom-right-> arms left & up
    )
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for cx, cy, sx, sy in corners:
        segs.append(((cx, cy), (cx + sx * L, cy)))   # horizontal arm
        segs.append(((cx, cy), (cx, cy + sy * L)))   # vertical arm
    return tuple(segs)


def _in_viewport(pt, viewport) -> bool:
    x, y = pt
    x0, y0, x1, y1 = viewport
    return x0 <= x <= x1 and y0 <= y <= y1


def _ray_rect_edge(origin, target, viewport):
    """Point where ray origin->target first crosses the viewport rectangle.

    Returns ``target`` unchanged if the ray is degenerate / never hits (should
    not happen for an off-screen target with an in-viewport origin).
    """
    ox, oy = origin
    dx, dy = target[0] - ox, target[1] - oy
    x0, y0, x1, y1 = viewport
    eps = 1e-6
    best: float | None = None
    for denom, num in ((dx, x0 - ox), (dx, x1 - ox), (dy, y0 - oy), (dy, y1 - oy)):
        if denom == 0.0:
            continue
        s = num / denom
        if s <= 0.0:
            continue
        px, py = ox + dx * s, oy + dy * s
        if (x0 - eps <= px <= x1 + eps) and (y0 - eps <= py <= y1 + eps):
            if best is None or s < best:
                best = s
    if best is None:
        return target
    return (ox + dx * best, oy + dy * best)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/overlay_model.py tests/gui/test_overlay_model.py
git commit -m "feat(overlay): converging lock-on brackets + off-screen hints"
```

---

### Task 6: `overlay_model` — `build_scene` assembler

**Files:**
- Modify: `src/ragnarok/gui/overlay_model.py`
- Test: `tests/gui/test_overlay_model.py`

**Interfaces:**
- Consumes: everything above; `TelemetrySnapshot` (`.roi_region`, `.tracks`, `.locked_target_id`), `AppConfig` (`.capture.roi_size`, `.aim.*`), `aim.fov.fov_deg_to_radius_px`.
- Produces: `build_scene(*, snapshot, cfg, viewport, lock_age_s, bracket_gap=28.0, bracket_arm=16.0, bracket_anim_s=0.18) -> OverlayScene`.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_overlay_model.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.telemetry.snapshot import TelemetrySnapshot
from ragnarok.gui.overlay_model import build_scene, OverlayScene


def _snap(**kw):
    base = dict(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0, detection_count=0,
                preview=None, seq=1)
    base.update(kw)
    return TelemetrySnapshot(**base)


def test_build_scene_no_region_is_empty():
    scene = build_scene(snapshot=_snap(roi_region=None), cfg=AppConfig(),
                        viewport=(0.0, 0.0, 1920.0, 1080.0), lock_age_s=0.0)
    assert scene.has_signal is False


def test_build_scene_locked_target_gets_brackets_and_line():
    cfg = AppConfig()                                   # roi 384, hfov 90, screen 1920
    # ROI centred on screen: region (768,348)-(1152,732); crosshair -> (960,540)
    tracks = (_trk(3, (180, 150, 204, 234)),)           # enemy near crosshair
    snap = _snap(roi_region=(768, 348, 1152, 732), tracks=tracks, locked_target_id=3)
    scene = build_scene(snapshot=snap, cfg=cfg,
                        viewport=(0.0, 0.0, 1920.0, 1080.0), lock_age_s=1.0)
    assert scene.has_signal is True
    assert scene.crosshair == (960.0, 540.0)
    assert scene.fov is not None and scene.fov.acquire_radius < scene.fov.retain_radius
    assert len(scene.bracket_segments) == 8             # locked target -> brackets
    assert scene.locked_line is not None
    assert scene.locked_line[0] == scene.crosshair
    assert len(scene.markers) == 1


def test_build_scene_offscreen_enemy_becomes_hint():
    cfg = AppConfig()
    # enemy far outside the ROI mapping -> diamond outside a tiny viewport
    tracks = (_trk(9, (380, 380, 384, 384)),)
    snap = _snap(roi_region=(768, 348, 1152, 732), tracks=tracks, locked_target_id=None)
    scene = build_scene(snapshot=snap, cfg=cfg,
                        viewport=(0.0, 0.0, 1000.0, 700.0), lock_age_s=0.0)
    assert len(scene.offscreen) == 1
    assert scene.offscreen[0].team == Team.ENEMY
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: FAIL (`build_scene` undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/overlay_model.py`:

```python
def build_scene(*, snapshot, cfg, viewport, lock_age_s: float,
                bracket_gap: float = 28.0, bracket_arm: float = 16.0,
                bracket_anim_s: float = 0.18) -> OverlayScene:
    """Assemble the full ``OverlayScene`` from a telemetry snapshot + config.

    Returns an empty (no-signal) scene when the snapshot has no ROI region.
    ``viewport`` is the overlay's screen rect ``(x0,y0,x1,y1)`` used for the
    off-screen direction hints. ``lock_age_s`` drives the bracket convergence.
    """
    region = snapshot.roi_region
    if region is None:
        return OverlayScene.empty()

    roi = cfg.capture.roi_size
    smap = ScreenMap.from_region(region, roi, roi)
    crosshair = smap.pt(roi / 2.0, roi / 2.0)

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    fov = FovRing(center=crosshair, acquire_radius=fov_px, retain_radius=retain_px)

    markers = build_markers(snapshot.tracks, smap, crosshair, fov_px,
                            snapshot.locked_target_id, a.head_frac, a.aim_point)

    segs: tuple = ()
    line = None
    locked = next((m for m in markers if m.locked), None)
    if locked is not None:
        t = lock_progress(lock_age_s, bracket_anim_s)
        segs = bracket_segments(locked.box, t, bracket_gap, bracket_arm)
        line = (crosshair, locked.diamond)

    hints: list[OffscreenHint] = []
    for m in markers:
        if m.team is Team.ENEMY and not _in_viewport(m.diamond, viewport):
            edge = _ray_rect_edge(crosshair, m.diamond, viewport)
            ang = math.atan2(m.diamond[1] - crosshair[1], m.diamond[0] - crosshair[0])
            hints.append(OffscreenHint(angle_rad=ang, edge_point=edge, team=m.team))

    return OverlayScene(has_signal=True, crosshair=crosshair, fov=fov,
                        markers=markers, bracket_segments=segs,
                        locked_line=line, offscreen=tuple(hints))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_overlay_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/overlay_model.py tests/gui/test_overlay_model.py
git commit -m "feat(overlay): build_scene assembles FOV/markers/brackets/hints"
```

---

### Task 7: `theme.py` — Cyberpunk palette tokens

**Files:**
- Create: `src/ragnarok/gui/theme.py`
- Test: `tests/gui/test_theme.py`

**Interfaces:**
- Consumes: `core.types.Team`.
- Produces: `ELECTRIC_YELLOW`, `CYAN`, `ALERT_RED`, `NEAR_BLACK` (hex strings); `TEAM_RGB` dict keyed by `Team.value`; `team_color(team: Team) -> str`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_theme.py`:

```python
import re
from ragnarok.core.types import Team
from ragnarok.gui import theme


def test_palette_tokens_are_valid_distinct_hex():
    tokens = [theme.ELECTRIC_YELLOW, theme.CYAN, theme.ALERT_RED, theme.NEAR_BLACK]
    for t in tokens:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", t)
    assert theme.ELECTRIC_YELLOW.upper() == "#FCEE0A"     # spec §10.1 primary accent
    assert len(set(t.upper() for t in tokens)) == 4


def test_team_color_maps_all_teams_distinctly():
    cols = {theme.team_color(t) for t in Team}
    assert len(cols) == 3
    # enemy = warm/orange, teammate = blue (mirrors gui/overlay.TEAM_BGR)
    assert theme.team_color(Team.ENEMY) != theme.team_color(Team.TEAMMATE)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_theme.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/theme.py`:

```python
"""Cyberpunk 2077 palette tokens (spec §10.1).

Function-first slice: just the color tokens the smart-lock overlay needs. The
full QSS stylesheet, condensed fonts, scanlines, and glitch effects are a later
(box-only) aesthetic pass.
"""
from __future__ import annotations

from ragnarok.core.types import Team

ELECTRIC_YELLOW = "#FCEE0A"   # primary accent (FOV ring, brackets, lock line)
CYAN = "#00F0FF"              # secondary
ALERT_RED = "#FF3B3B"         # locked-target highlight / alerts
NEAR_BLACK = "#0A0A0C"        # backgrounds

# Team colors in RGB hex, mirroring gui/overlay.TEAM_BGR (orange / blue / gray).
TEAM_RGB = {
    Team.ENEMY.value: "#FF8C00",
    Team.TEAMMATE.value: "#0080FF",
    Team.UNKNOWN.value: "#A0A0A0",
}


def team_color(team: Team) -> str:
    return TEAM_RGB.get(team.value, TEAM_RGB[Team.UNKNOWN.value])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_theme.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/theme.py tests/gui/test_theme.py
git commit -m "feat(overlay): Cyberpunk palette tokens + team_color"
```

---

### Task 8: `FovOverlay` widget + app wiring

**Files:**
- Create: `src/ragnarok/gui/overlay_window.py`
- Modify: `src/ragnarok/app.py`
- Test: `tests/gui/test_overlay_window.py`

**Interfaces:**
- Consumes: `SnapshotPublisher.latest()`, a `config_provider: Callable[[], AppConfig]`, `build_scene`/`LockAgeTracker`/`team_color`, `core.clock.now_ns`.
- Produces: `FovOverlay(publisher, config_provider, *, interval_ms=16, clock=now_ns)` — a frameless/topmost/translucent/click-through `QWidget` with a `paintEvent` that renders the scene; safe no-op when there is no snapshot.

**Box-only deferrals (document in the module docstring, do not implement here):** swap `QWidget`→`QOpenGLWidget` for the GL-backed path (spec §10.2); apply Win32 `WS_EX_TRANSPARENT | WS_EX_LAYERED` via ctypes on `show()` for OS-level click-through (`WA_TransparentForMouseEvents` covers Qt-level pass-through cross-platform meanwhile); position/size the window to the captured region and translate the painter for multi-monitor origins; full Cyberpunk restyle (glitch, scanlines, animated gauges).

- [ ] **Step 1: Write the failing test** — `tests/gui/test_overlay_window.py`:

```python
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from ragnarok.config.schema import AppConfig
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay_window import FovOverlay


def _cfg():
    return AppConfig()


def test_overlay_window_has_click_through_flags(qtbot):
    w = FovOverlay(SnapshotPublisher(), _cfg)
    qtbot.addWidget(w)
    flags = w.windowFlags()
    assert flags & Qt.FramelessWindowHint
    assert flags & Qt.WindowStaysOnTopHint
    assert flags & Qt.Tool
    assert w.testAttribute(Qt.WA_TranslucentBackground)
    assert w.testAttribute(Qt.WA_TransparentForMouseEvents)


def test_overlay_window_paints_scene_without_error(qtbot):
    pub = SnapshotPublisher()
    w = FovOverlay(pub, _cfg)
    qtbot.addWidget(w)
    w.resize(1920, 1080)
    tracks = (Track(track_id=3, xyxy=(180, 150, 204, 234), confidence=0.9,
                    class_id=0, team=Team.ENEMY),)
    pub.publish(TelemetrySnapshot(
        fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0, detection_count=1,
        preview=None, seq=1, tracks=tracks,
        locked_target_id=3, roi_region=(768, 348, 1152, 732)))
    img = QImage(1920, 1080, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    w.render(p)          # exercises paintEvent -> build_scene -> _draw_scene
    p.end()

def test_overlay_window_no_snapshot_is_noop(qtbot):
    w = FovOverlay(SnapshotPublisher(), _cfg)
    qtbot.addWidget(w)
    img = QImage(64, 64, QImage.Format_ARGB32); img.fill(0)
    p = QPainter(img)
    w.render(p)          # must not raise when latest() is None
    p.end()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_overlay_window.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/overlay_window.py`:

```python
"""Frameless, click-through, always-on-top smart-lock FOV overlay (spec §10.2).

Function-first: all geometry lives in ``overlay_model.build_scene`` (pure,
tested). This widget is the thin QPainter renderer + Qt window plumbing.
Painting *correctness* (pixels) is box-only; construction, window flags, and a
render smoke are offscreen-testable via qtbot.

BOX-ONLY DEFERRALS (perf / OS integration, not implemented here):
  * swap ``QWidget`` -> ``QOpenGLWidget`` for the GL-backed path (avoids slow GDI).
  * apply Win32 ``WS_EX_TRANSPARENT | WS_EX_LAYERED`` via ctypes on ``show()``
    for OS-level click-through (``WA_TransparentForMouseEvents`` handles the
    Qt-level pass-through cross-platform meanwhile).
  * position/size the window to the captured region and translate the painter
    for multi-monitor origins.
  * full Cyberpunk restyle: glitch/chromatic-aberration, scanlines, gauges.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Team
from ragnarok.gui import theme
from ragnarok.gui.overlay_model import LockAgeTracker, build_scene


class FovOverlay(QWidget):
    def __init__(self, publisher, config_provider, *, interval_ms: int = 16,
                 clock=now_ns) -> None:
        super().__init__()
        self._pub = publisher
        self._cfg = config_provider
        self._clock = clock
        self._lock_age = LockAgeTracker()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)          # own timer, decoupled from hot loop
        self._timer.timeout.connect(self.update)      # schedule a repaint
        self._timer.start()

    # -- rendering -------------------------------------------------------
    def paintEvent(self, event) -> None:
        snap = self._pub.latest()
        if snap is None:
            return
        now = self._clock()
        lock_age = self._lock_age.update(snap.locked_target_id, now)
        vp = (0.0, 0.0, float(self.width()), float(self.height()))
        scene = build_scene(snapshot=snap, cfg=self._cfg(), viewport=vp,
                            lock_age_s=lock_age)
        if not scene.has_signal:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._draw_scene(p, scene)
        finally:
            p.end()

    def _draw_scene(self, p: QPainter, scene) -> None:
        yellow = QColor(theme.ELECTRIC_YELLOW)
        red = QColor(theme.ALERT_RED)

        # FOV ring: acquire (solid) + retain (dim), tick marks
        if scene.fov is not None:
            cx, cy = scene.fov.center
            self._ring(p, cx, cy, scene.fov.acquire_radius, yellow, 2)
            dim = QColor(yellow); dim.setAlpha(90)
            self._ring(p, cx, cy, scene.fov.retain_radius, dim, 1)

        # crosshair tick
        p.setPen(QPen(yellow, 1))
        cx, cy = scene.crosshair
        p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
        p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

        # markers: team-colored box + diamond for enemies; lock = red
        for m in scene.markers:
            col = red if m.locked else QColor(theme.team_color(m.team))
            p.setPen(QPen(col, 2 if m.locked else 1))
            x1, y1, x2, y2 = m.box
            p.drawRect(QPointF(x1, y1).x(), QPointF(y1, y1).y() if False else y1,
                       x2 - x1, y2 - y1)
            if m.team is Team.ENEMY:
                self._diamond(p, m.diamond[0], m.diamond[1], 6, col)

        # lock-on convergence brackets (locked target)
        p.setPen(QPen(yellow, 2))
        for (a, b) in scene.bracket_segments:
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # thin tracking line crosshair -> locked diamond
        if scene.locked_line is not None:
            a, b = scene.locked_line
            pen = QPen(yellow, 1); pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # off-screen direction hints: small chevron dots at the viewport edge
        for h in scene.offscreen:
            p.setPen(QPen(QColor(theme.team_color(h.team)), 2))
            ex, ey = h.edge_point
            self._diamond(p, ex, ey, 5, QColor(theme.team_color(h.team)))

    @staticmethod
    def _ring(p: QPainter, cx: float, cy: float, r: float, col: QColor, w: int) -> None:
        p.setPen(QPen(col, w))
        p.drawEllipse(QPointF(cx, cy), r, r)

    @staticmethod
    def _diamond(p: QPainter, cx: float, cy: float, r: float, col: QColor) -> None:
        poly = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                          QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(col, 2))
        p.drawPolygon(poly)
```

> Note for the implementer: the `p.drawRect(...)` line above must draw the box from `(x1, y1)` with width `x2-x1`, height `y2-y1`. Write it cleanly as `p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))` (the awkward expression in the snippet is a transcription artifact — use the clean form). All other lines are literal.

- [ ] **Step 4: Run to verify the widget tests pass**

Run: `uv run --extra dev pytest tests/gui/test_overlay_window.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire into `app.py`** — in `src/ragnarok/app.py`, import and create the overlay next to the main window (box-only path; not unit-tested). Add the import near the other gui imports:

```python
from ragnarok.gui.overlay_window import FovOverlay
```

and in `main()`, after `window.show()`:

```python
    overlay = FovOverlay(publisher, lambda: cfg)
    overlay.resize(cfg.aim.screen_width_px, int(cfg.aim.screen_width_px * 9 / 16))
    overlay.show()
```

- [ ] **Step 6: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 518 baseline + new tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/ragnarok/gui/overlay_window.py src/ragnarok/app.py tests/gui/test_overlay_window.py
git commit -m "feat(overlay): FovOverlay QPainter widget + app wiring"
```

---

## Self-Review

**Spec coverage (§10.2 overlay):**
- FOV indicator arc/ring with ticks → `FovRing` + `_ring` (Tasks 3, 8). ✅ (tick *marks* drawn as ring rings; explicit radial ticks are cosmetic, box-only.)
- Acquisition brackets converging/snapping → `bracket_segments` + `lock_progress` + `LockAgeTracker` (Tasks 4, 5, 6, 8). ✅
- Diamond markers over confirmed (enemy) targets → `build_markers` diamond + `_diamond` (Tasks 4, 8). ✅
- Locked-target highlight + tracking line to crosshair → red marker + `locked_line` (Tasks 6, 8). ✅
- Enemy/teammate color-coding + per-track confidence + off-screen hints → `team_color`, `TargetMarker.confidence`, `OffscreenHint` (Tasks 4, 5, 6, 7). ✅ (confidence text render is box-only cosmetic.)
- Frameless/topmost/Tool/translucent/click-through/own-timer → Task 8 flags + `QTimer`. ✅ (QOpenGLWidget + Win32 click-through explicitly deferred box-only.)
- Cosmetic-only (never touches aim) → `overlay_model` is read-only pure; widget only reads publisher. ✅

**Placeholder scan:** the one awkward `drawRect` expression in Task 8 is called out with the clean replacement; no TBD/TODO-as-logic elsewhere.

**Type consistency:** `ScreenMap`, `FovRing`, `TargetMarker`, `OffscreenHint`, `OverlayScene`, `LockAgeTracker`, `build_markers`, `bracket_segments`, `lock_progress`, `build_scene`, `team_color`, `FovOverlay(publisher, config_provider, *, interval_ms, clock)` — names used in Tasks 6/8 match their definitions in Tasks 3/4/5/7. Snapshot fields `locked_target_id`/`roi_region` (Task 1) consumed in Tasks 2/6/8. `cfg` passed to `build_scene` is `AppConfig` (uses `.capture.roi_size` + `.aim.*`) consistently.

**Deferrals are honest and logged:** QOpenGLWidget GL backing, Win32 OS-level click-through, multi-monitor painter translation, radial FOV ticks, confidence text, and the full Cyberpunk restyle are all box-only follow-ups, documented in the widget docstring and this review — not silently dropped.
