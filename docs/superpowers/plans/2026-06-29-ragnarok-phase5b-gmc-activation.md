# Ragnarok Phase 5B — Feed-Forward GMC Activation & Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the already-built `FeedForwardGMC` in the live tracking loop (it currently sits on a dead `IDENTITY_AFFINE` path) by sharing one commanded-motion buffer between the `AimController` (producer) and the tracker's ego-motion provider (consumer), and add the pure calibration math (τ_render lag + signed `deg_per_count`) the GMC needs to be trusted.

**Architecture:** The cleanest activation needs **no vendored-core edit and has no double-apply hazard**: the vendored BoT-SORT core already computes its global-motion warp from a single injected `ego.estimate(frame)` provider. So we (1) make the tracker forward the real `Frame` to the core, (2) inject `FeedForwardGMC` (sharing the `AimController`'s `CommandedMotionBuffer`) as that ego provider when configured, and (3) leave identity as the default. The GMC back-projects our own commanded mouse counts into a 2×3 affine over a τ_render-aligned window; calibration of `deg_per_count`/`τ_render` is provided as pure, unit-tested solvers (the live optical-flow capture that feeds them is a box-only smoke).

**Tech Stack:** Python 3.11+, numpy, scipy (existing — reuses `diagnostics/resample.py` for the τ_render cross-correlation). No new third-party dependencies. The vendored MIT BoT-SORT core is **not modified**.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is tracking/control-systems engineering (spec §5.3, §6.4, §18).
- **No vendored-core edit:** `src/ragnarok/tracking/_vendor/` is MIT-vendored and must not change. Activation works entirely through the existing `ego` injection seam + forwarding the `Frame`.
- **No double-apply:** exactly one source computes the GMC warp — the injected `ego` provider. The tracker must not also apply a second affine.
- **CI-safe always:** no GPU/display/cursor/game in unit tests. The GMC math (`FeedForwardGMC`, `CommandedMotionBuffer`) is already pure + tested. The calibration solvers are pure numpy; the live optical-flow / in-game capture that produces their inputs is a **box-only smoke**, behind the function boundary.
- **Integer-ns time math** (perf_counter_ns); float-seconds only inside the calibration/resample layer.
- **Shared buffer, single producer:** the `AimController` pushes commanded counts (`sx*k`, already implemented) into ONE `CommandedMotionBuffer`; the tracker's `FeedForwardGMC` reads that SAME object. App wiring owns its construction.
- **Backward compatibility:** GMC defaults **off** (`tracking.gmc = "off"` → `IdentityEgoMotion`); identity tracking and all existing tests stay green. Frozen pydantic config, backward-compatible TOML round-trip.
- **`deg_per_count` is empirical and SIGNED** (sign + magnitude calibrated against the game); `AimConfig.sensitivity` is `gt=0` and unrelated, so the signed value lives in `TrackingConfig`.
- **TDD, frequent commits, exact file paths.** Match codebase idiom (`from __future__ import annotations`, keyword-only constructors, module docstrings, focused files).

## Scope Boundary (explicit deferrals)

- **Live optical-flow τ_render auto-estimation + the in-game calibration wizard** (capturing global flow via `cv2.phaseCorrelate`/Farnebäck while commanding motion) → **box-only**. This plan ships the pure lag/`deg_per_count` solvers that consume already-captured series; the live capture loop that produces those series is a manual-smoke driver (documented, not in CI).
- **Physical-mouse passthrough deltas** (USB Host Shield) → Phase 7. The shared `CommandedMotionBuffer` already accepts additional pushes, so passthrough only adds a second producer later.
- **Recoil-into-ego-motion integrator** (§6.4/§6.6) → later; recoil stays an additive output offset for now.
- **World-angular target filtering** (§6.4 root fix) → later; this plan delivers the pixel-space active-GMC predict warp (the association-side fix), which is the prerequisite.
- **CV-GMC (optical-flow ORB/ECC) fallback** → not needed; feed-forward GMC is the default and the only provider this plan wires.

---

## File Structure

**New files:**
- `src/ragnarok/tracking/calibration.py` — pure calibration solvers: `estimate_tau_render`, `solve_deg_per_count`.
- `tests/tracking/test_calibration.py`
- `tests/tracking/test_gmc_activation.py` — the activation wiring integration test.

**Modified files:**
- `src/ragnarok/tracking/base.py` — `Tracker.update(detections, frame=None)` (replace the vestigial `ego_affine` param; keep `IDENTITY_AFFINE` defined); `IdentityTracker` ignores `frame`.
- `src/ragnarok/tracking/botsort.py` — `BotSortTracker.update(detections, frame=None)` forwards the real frame to the core (`self._core.update(rows, frame=frame)`); expose `.ego`.
- `src/ragnarok/worker/loop.py` — pass `frame` to `self._tracker.update(dets, frame)` (drop the `IDENTITY_AFFINE` positional).
- `src/ragnarok/config/schema.py` — `TrackingConfig` gains `gmc`, `deg_per_count`, `tau_render_s`.
- `src/ragnarok/wiring.py` — `build_tracker(cfg, *, gmc_buffer=None)` injects `FeedForwardGMC` as the ego when `gmc == "feedforward"`.
- `src/ragnarok/app.py` — create ONE `CommandedMotionBuffer`; pass it to both `build_tracker(gmc_buffer=...)` and `_build_aim_controller(..., commanded_buffer=...)`.
- `tests/worker/test_loop.py` — update the `_Trk` fake signature to `update(self, detections, frame=None)`.

---

## Task 1: Tracker.update(detections, frame=None) — forward the Frame to the GMC

**Files:**
- Modify: `src/ragnarok/tracking/base.py`, `src/ragnarok/tracking/botsort.py`, `src/ragnarok/worker/loop.py`, `tests/worker/test_loop.py`
- Test: `tests/tracking/test_botsort.py` (extend)

**Interfaces:**
- Consumes: the vendored core's `BoTSORT.update(output_results, frame)` which computes `warp = self.ego.estimate(frame)`.
- Produces: `Tracker.update(self, detections: Detections, frame=None) -> Tracks` on the ABC, `IdentityTracker` (ignores `frame`), and `BotSortTracker` (forwards `frame` to the core so its injected `ego` sees the real `Frame.t_capture_ns`). `BotSortTracker.ego` is a public attribute (already set in `__init__`). `IDENTITY_AFFINE` stays defined in `base.py` (unchanged). The worker calls `tracker.update(dets, frame)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_botsort.py
def test_update_forwards_frame_to_ego():
    import numpy as np
    from ragnarok.tracking.botsort import BotSortTracker
    from ragnarok.tracking.egomotion import EgoMotion

    class _SpyEgo(EgoMotion):
        def __init__(self):
            self.seen = "unset"
        def estimate(self, frame):
            self.seen = frame                     # capture what the core passed
            return np.eye(2, 3, dtype=np.float32)

    spy = _SpyEgo()
    tr = BotSortTracker(ego=spy)
    sentinel = object()
    tr.update(_dets((10, 10, 30, 50)), sentinel)  # 2nd positional arg is `frame`
    assert tr.ego is spy
    assert spy.seen is sentinel                    # the exact frame reached the ego
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_botsort.py::test_update_forwards_frame_to_ego -v`
Expected: FAIL — the core is called with `frame=None`, so `spy.seen` is `None`, not the sentinel.

- [ ] **Step 3: Change the interface and forward the frame**

In `src/ragnarok/tracking/base.py`, change the ABC and `IdentityTracker` (leave `IDENTITY_AFFINE` defined above them):

```python
class Tracker(ABC):
    """Maps detections to tracks with stable ids across frames."""

    @abstractmethod
    def update(self, detections: Detections, frame=None) -> Tracks:
        """Advance one frame and return the current confirmed tracks.

        ``frame`` is the captured Frame (or None); a GMC ego provider reads its
        ``t_capture_ns`` to build the global-motion warp. Trackers that don't do
        ego-motion compensation ignore it.
        """
        raise NotImplementedError
```

```python
class IdentityTracker(Tracker):
    # ... docstring unchanged ...
    def update(self, detections: Detections, frame=None) -> Tracks:
        tracks = []
        for det in detections:
            self._next_id += 1
            tracks.append(Track.from_detection(det, self._next_id))
        return Tracks(items=tuple(tracks))
```

In `src/ragnarok/tracking/botsort.py`, change `update` to forward the frame:

```python
    def update(self, detections: Detections, frame=None) -> Tracks:
        rows = [
            [d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3], d.confidence, d.class_id]
            for d in detections
        ]
        output_results = (
            np.asarray(rows, dtype=float) if rows else np.empty((0, 6), dtype=float)
        )
        # The frame is consumed only by the injected ego provider
        # (IdentityEgoMotion ignores it; FeedForwardGMC reads frame.t_capture_ns).
        stracks = self._core.update(output_results, frame=frame)
        # ... rest unchanged (frame_id, Track construction) ...
```

In `src/ragnarok/worker/loop.py`, update the import and the call:

```python
# change the import line
from ragnarok.tracking.base import Tracker, IdentityTracker
# ... in tick(), replace the tracker call:
        tracks = self._tracker.update(dets, frame)   # frame carries t_capture_ns for GMC
```

In `tests/worker/test_loop.py`, update the `_Trk` fake:

```python
    class _Trk:
        def update(self, detections, frame=None):
            return Tracks(items=(Track(track_id=42, xyxy=(0, 0, 10, 10),
                                       confidence=0.9, class_id=0),))
```

- [ ] **Step 4: Run the touched suites to verify they pass**

Run: `python -m pytest tests/tracking tests/worker -q`
Expected: PASS (the new spy test; existing tracking tests call `update(dets)` with no 2nd arg; the worker tests now use the `frame=` fake).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/tracking/base.py src/ragnarok/tracking/botsort.py src/ragnarok/worker/loop.py tests/worker/test_loop.py tests/tracking/test_botsort.py
git commit -m "refactor(tracking): Tracker.update(detections, frame) — forward frame to the ego provider"
```

---

## Task 2: TrackingConfig GMC fields

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Test: `tests/config/test_tracking_config.py` (extend)

**Interfaces:**
- Produces: `TrackingConfig` gains `gmc: Literal["off", "feedforward"] = "off"`, `deg_per_count: float = 0.0` (SIGNED — may be negative; `0.0` = uncalibrated/no-op warp), `tau_render_s: float = Field(default=0.0, ge=0.0, le=0.1)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/config/test_tracking_config.py
def test_gmc_defaults_off():
    assert TrackingConfig().gmc == "off"
    assert TrackingConfig().deg_per_count == 0.0
    assert TrackingConfig().tau_render_s == 0.0


def test_gmc_feedforward_and_signed_deg_per_count():
    t = TrackingConfig(gmc="feedforward", deg_per_count=-0.022, tau_render_s=0.012)
    assert t.gmc == "feedforward"
    assert t.deg_per_count == -0.022          # signed: negative is valid
    assert t.tau_render_s == 0.012


def test_gmc_rejects_unknown_mode():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrackingConfig(gmc="optical")  # type: ignore[arg-type]


def test_tau_render_bounds():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrackingConfig(tau_render_s=-0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_tracking_config.py -k gmc -v`
Expected: FAIL — `TrackingConfig` has no `gmc` field.

- [ ] **Step 3: Add the fields**

In `src/ragnarok/config/schema.py`, add to `TrackingConfig` (after `proximity_thresh`):

```python
    # --- Phase 5B feed-forward GMC ---
    gmc: Literal["off", "feedforward"] = "off"
    deg_per_count: float = 0.0          # SIGNED degrees of yaw/pitch per mouse count (empirical)
    tau_render_s: float = Field(default=0.0, ge=0.0, le=0.1)   # render+display latency window
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config -q`
Expected: PASS (incl. the TOML round-trip in `test_store.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py tests/config/test_tracking_config.py
git commit -m "feat(config): TrackingConfig GMC fields (mode, signed deg_per_count, tau_render)"
```

---

## Task 3: Wire FeedForwardGMC into the tracker via a shared buffer

**Files:**
- Modify: `src/ragnarok/wiring.py`, `src/ragnarok/app.py`
- Test: `tests/test_wiring.py` (extend)

**Interfaces:**
- Consumes: `FeedForwardGMC`, `CommandedMotionBuffer` (`ragnarok.tracking.egomotion`); `BotSortTracker(ego=...)`; `TrackingConfig.gmc/deg_per_count/tau_render_s` (T2); `AimConfig.hfov_deg/screen_width_px`; `CaptureConfig.target_fps`.
- Produces:
  - `build_tracker(cfg: AppConfig, *, gmc_buffer=None) -> Tracker` — when `cfg.tracking.gmc == "feedforward"` and `gmc_buffer is not None` and the backend is `botsort`, construct `FeedForwardGMC(hfov_deg=cfg.aim.hfov_deg, screen_width_px=cfg.aim.screen_width_px, deg_per_count=cfg.tracking.deg_per_count, tau_render_s=cfg.tracking.tau_render_s, frame_dt_s=1.0/cfg.capture.target_fps, buffer=gmc_buffer)` and pass it as `BotSortTracker(ego=...)`. Otherwise behave exactly as before (identity ego). `identity` backend ignores GMC.
  - `app._build_aim_controller(cfg, commanded_buffer)` — accept the shared buffer instead of creating its own.
  - `app.main()` — create one `CommandedMotionBuffer`, pass it to both `build_tracker(gmc_buffer=...)` and `_build_aim_controller(..., commanded_buffer=...)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_wiring.py
def test_build_tracker_no_gmc_by_default_uses_identity_ego():
    from ragnarok.wiring import build_tracker
    from ragnarok.tracking.egomotion import IdentityEgoMotion
    from ragnarok.tracking.botsort import BotSortTracker
    trk = build_tracker(AppConfig(), gmc_buffer=object())  # gmc off by default
    assert isinstance(trk, BotSortTracker)
    assert isinstance(trk.ego, IdentityEgoMotion)


def test_build_tracker_feedforward_injects_shared_buffer_gmc():
    from ragnarok.wiring import build_tracker
    from ragnarok.tracking.egomotion import CommandedMotionBuffer, FeedForwardGMC
    buf = CommandedMotionBuffer()
    cfg = AppConfig(tracking={"gmc": "feedforward", "deg_per_count": 0.02})
    trk = build_tracker(cfg, gmc_buffer=buf)
    assert isinstance(trk.ego, FeedForwardGMC)
    assert trk.ego.buffer is buf                    # SAME buffer object (shared)


def test_build_tracker_feedforward_without_buffer_stays_identity():
    # No buffer supplied -> cannot share -> fall back to identity ego (safe).
    from ragnarok.wiring import build_tracker
    from ragnarok.tracking.egomotion import IdentityEgoMotion
    cfg = AppConfig(tracking={"gmc": "feedforward", "deg_per_count": 0.02})
    trk = build_tracker(cfg, gmc_buffer=None)
    assert isinstance(trk.ego, IdentityEgoMotion)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wiring.py -k "gmc or feedforward or ego" -v`
Expected: FAIL — `build_tracker` has no `gmc_buffer` kwarg.

- [ ] **Step 3: Implement the wiring**

In `src/ragnarok/wiring.py`, replace `build_tracker` with:

```python
def build_tracker(cfg: AppConfig, *, gmc_buffer=None) -> Tracker:
    t = cfg.tracking
    if t.backend == "identity":
        return IdentityTracker()
    from ragnarok.tracking.botsort import BotSortTracker
    ego = None
    if t.gmc == "feedforward" and gmc_buffer is not None:
        from ragnarok.tracking.egomotion import FeedForwardGMC
        ego = FeedForwardGMC(
            hfov_deg=cfg.aim.hfov_deg, screen_width_px=cfg.aim.screen_width_px,
            deg_per_count=t.deg_per_count, tau_render_s=t.tau_render_s,
            frame_dt_s=1.0 / cfg.capture.target_fps, buffer=gmc_buffer,
        )
    return BotSortTracker(
        ego=ego,                                  # None -> BotSortTracker uses IdentityEgoMotion
        track_high_thresh=t.track_high_thresh,
        track_low_thresh=t.track_low_thresh,
        new_track_thresh=t.new_track_thresh,
        track_buffer=t.track_buffer,
        match_thresh=t.match_thresh,
        proximity_thresh=t.proximity_thresh,
        frame_rate=cfg.capture.target_fps,
    )
```

In `src/ragnarok/app.py`, change `_build_aim_controller` to accept the buffer and `main()` to share it. Replace the `CommandedMotionBuffer()` line inside `_build_aim_controller` (it currently constructs its own) so the function takes it as a parameter:

```python
def _build_aim_controller(cfg, commanded_buffer):
    # ... existing imports/builders ...
    # (delete the local `CommandedMotionBuffer()` construction)
    return AimController(
        a, selector=selector, imm_manager=IMMManager(),
        aimer=build_aimer(cfg), mouse=mouse, is_aim_active=is_active,
        roi_size=cfg.capture.roi_size,
        shaper=build_shaper(cfg),
        vel_smoother=VelocitySmoother(alpha=a.vel_smooth_alpha, max_px_s=a.vel_clamp_px_s),
        adaptive_lead=AdaptiveLead(alpha=a.lead_alpha, base_latency_s=a.lead_ms / 1000.0),
        recoil=build_recoil(cfg),
        trigger=trigger, trigger_active=trigger_active,
        commanded_buffer=commanded_buffer,
    )
```

And in `main()`:

```python
    from ragnarok.tracking.egomotion import CommandedMotionBuffer
    cmd_buffer = CommandedMotionBuffer()
    aim_controller = _build_aim_controller(cfg, cmd_buffer) if cfg.aim.enabled else None
    loop = WorkerLoop(
        create_capturer(cfg.capture), create_detector(cfg.detection),
        StageProfiler(), publisher,
        tracker=build_tracker(cfg, gmc_buffer=cmd_buffer),
        classifier=build_classifier(cfg), aim_controller=aim_controller,
    )
```

(Keep the existing `CommandedMotionBuffer` import out of `_build_aim_controller`; it is imported in `main()` now.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wiring.py -q`
Expected: PASS

- [ ] **Step 5: Verify app builds under offscreen Qt**

Run:
```bash
QT_QPA_PLATFORM=offscreen python -c "import ragnarok.app as a; from ragnarok.config.schema import AppConfig; cfg=AppConfig(tracking={'gmc':'feedforward','deg_per_count':0.02}); from ragnarok.tracking.egomotion import CommandedMotionBuffer; b=CommandedMotionBuffer(); t=a.build_tracker(cfg, gmc_buffer=b); print(type(t.ego).__name__, t.ego.buffer is b)"
```
Expected: prints `FeedForwardGMC True`.

- [ ] **Step 6: Run the full suite + commit**

Run: `python -m pytest -q`
Expected: PASS

```bash
git add src/ragnarok/wiring.py src/ragnarok/app.py tests/test_wiring.py
git commit -m "feat(app): activate FeedForwardGMC via a shared CommandedMotionBuffer (off by default)"
```

---

## Task 4: τ_render cross-correlation lag solver (pure)

**Files:**
- Create: `src/ragnarok/tracking/calibration.py`
- Create: `tests/tracking/test_calibration.py`

**Interfaces:**
- Consumes: nothing (numpy).
- Produces: `estimate_tau_render(commanded, measured, dt_s, *, max_lag_s=0.1) -> float` — given two equal-length, already-uniform 1-D series (`commanded` = the commanded motion signal, `measured` = the observed global-motion/optical-flow signal on the same uniform grid and `dt_s` spacing), return the lag in seconds (≥ 0, capped at `max_lag_s`) at which shifting `measured` back by that lag best correlates with `commanded` — i.e. the render+display latency by which the screen response trails the command. (The caller resamples raw `(t_ns, value)` series to a common grid via `diagnostics.resample.resample_uniform` first; the live optical-flow capture that produces `measured` is a box-only smoke.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_calibration.py
"""Tests for the pure GMC calibration solvers."""
from __future__ import annotations
import numpy as np
from ragnarok.tracking.calibration import estimate_tau_render


def test_recovers_known_lag():
    dt = 0.001
    n = 500
    rng = np.zeros(n)
    rng[100:140] = 1.0                       # a commanded pulse
    commanded = rng
    measured = np.zeros(n)
    measured[115:155] = 1.0                  # same pulse delayed by 15 samples = 15 ms
    lag = estimate_tau_render(commanded, measured, dt, max_lag_s=0.1)
    assert abs(lag - 0.015) < 1.5e-3


def test_zero_lag_when_aligned():
    dt = 0.001
    x = np.zeros(200)
    x[50:90] = 1.0
    lag = estimate_tau_render(x, x.copy(), dt)
    assert abs(lag) < 1e-9


def test_lag_capped_at_max():
    dt = 0.001
    n = 400
    commanded = np.zeros(n); commanded[10:30] = 1.0
    measured = np.zeros(n); measured[300:320] = 1.0   # 290 ms apart, beyond max
    lag = estimate_tau_render(commanded, measured, dt, max_lag_s=0.05)
    assert lag <= 0.05 + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_calibration.py -v`
Expected: FAIL — `No module named 'ragnarok.tracking.calibration'`.

- [ ] **Step 3: Implement estimate_tau_render**

```python
# src/ragnarok/tracking/calibration.py
"""Pure feed-forward-GMC calibration solvers (spec §5.3, §18).

estimate_tau_render: the render+display latency by which the on-screen response
trails a commanded motion, found by cross-correlating the commanded signal with
the measured global-motion (optical-flow) signal on a common uniform grid. The
live optical-flow capture that produces `measured` is a box-only smoke; this
function (the analysis) is pure and unit-tested.
"""
from __future__ import annotations

import numpy as np


def estimate_tau_render(commanded, measured, dt_s: float, *, max_lag_s: float = 0.1) -> float:
    c = np.asarray(commanded, dtype=float)
    m = np.asarray(measured, dtype=float)
    c = c - c.mean()
    m = m - m.mean()
    # Full cross-correlation; positive lag = measured trails commanded.
    corr = np.correlate(m, c, mode="full")
    lags = np.arange(-(len(c) - 1), len(m))          # sample lags aligned with corr
    max_lag = int(round(max_lag_s / dt_s))
    keep = (lags >= 0) & (lags <= max_lag)           # render latency is non-negative
    if not keep.any():
        return 0.0
    best = lags[keep][int(np.argmax(corr[keep]))]
    return float(best) * dt_s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/tracking/calibration.py tests/tracking/test_calibration.py
git commit -m "feat(tracking): tau_render cross-correlation lag solver (pure)"
```

---

## Task 5: Signed deg_per_count solver (pure)

**Files:**
- Modify: `src/ragnarok/tracking/calibration.py`
- Test: `tests/tracking/test_calibration.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `solve_deg_per_count(total_counts: float, measured_total_deg: float) -> float` — from a calibration turn (command a known total horizontal mouse counts across a static reference, measure the total angular displacement the world rotated in degrees), return the SIGNED `deg_per_count = measured_total_deg / total_counts`. Raises `ValueError` if `total_counts == 0`. Sign is preserved (a leftward measured rotation under positive commanded counts yields a negative ratio — the value the GMC needs).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/tracking/test_calibration.py
from ragnarok.tracking.calibration import solve_deg_per_count
import pytest


def test_deg_per_count_basic():
    assert abs(solve_deg_per_count(1000.0, 22.0) - 0.022) < 1e-9


def test_deg_per_count_preserves_sign():
    # positive commanded counts, world rotated the other way -> negative ratio
    assert solve_deg_per_count(1000.0, -22.0) < 0.0


def test_deg_per_count_zero_counts_raises():
    with pytest.raises(ValueError):
        solve_deg_per_count(0.0, 22.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_calibration.py -k deg_per_count -v`
Expected: FAIL — `cannot import name 'solve_deg_per_count'`.

- [ ] **Step 3: Implement solve_deg_per_count**

Append to `src/ragnarok/tracking/calibration.py`:

```python
def solve_deg_per_count(total_counts: float, measured_total_deg: float) -> float:
    """Signed degrees of world rotation per commanded mouse count (spec §18).

    From a calibration turn across a static reference: command total_counts and
    measure the total angular displacement (deg) the world rotated. Sign is
    preserved so the GMC back-projection uses the correct direction.
    """
    if total_counts == 0.0:
        raise ValueError("total_counts must be non-zero to solve deg_per_count")
    return measured_total_deg / total_counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/tracking/calibration.py tests/tracking/test_calibration.py
git commit -m "feat(tracking): signed deg_per_count solver (pure)"
```

---

## Task 6: GMC activation integration test (shared buffer → non-identity warp)

**Files:**
- Create: `tests/tracking/test_gmc_activation.py`

**Interfaces:**
- Consumes: `build_tracker` (T3), `CommandedMotionBuffer`/`FeedForwardGMC` (existing), `Frame` (`ragnarok.core.types`), `focal_length_px` (`ragnarok.aim.fov`).
- Produces: a CI-safe integration test proving activation is real end-to-end: the assembled tracker's ego is a `FeedForwardGMC` sharing the controller's buffer, and after a commanded pan is pushed into that buffer, `tracker.ego.estimate(frame)` returns a non-identity affine in the correct (back-projected) direction. (The full "residual collapse during a live turn across a static target" is the box-only acceptance smoke — documented here, not in CI.)

- [ ] **Step 1: Write the test (this is the deliverable)**

```python
# tests/tracking/test_gmc_activation.py
"""Integration: a commanded pan in the shared buffer produces a real GMC warp.

Proves the Phase 5B activation wiring end-to-end without a GPU/game: the tracker
built with gmc='feedforward' shares the AimController's CommandedMotionBuffer,
and a pushed commanded pan yields a non-identity, correctly-signed warp. The full
residual-collapse validation against a live static target is a box-only smoke.
"""
from __future__ import annotations
import math
import numpy as np
from ragnarok.config.schema import AppConfig
from ragnarok.wiring import build_tracker
from ragnarok.tracking.egomotion import CommandedMotionBuffer, FeedForwardGMC
from ragnarok.core.types import Frame
from ragnarok.aim.fov import focal_length_px


def test_commanded_pan_produces_non_identity_warp_through_shared_buffer():
    buf = CommandedMotionBuffer()
    cfg = AppConfig(
        aim={"hfov_deg": 90.0, "screen_width_px": 1920},
        capture={"target_fps": 100},
        tracking={"gmc": "feedforward", "deg_per_count": 0.02, "tau_render_s": 0.0},
    )
    tracker = build_tracker(cfg, gmc_buffer=buf)
    assert isinstance(tracker.ego, FeedForwardGMC)
    assert tracker.ego.buffer is buf

    t_cap = 1_000_000_000
    # Push a commanded rightward pan inside the GMC window [t_cap - frame_dt, t_cap].
    buf.push(t_cap - 5_000_000, 100.0, 0.0)        # frame_dt = 1/100 s = 10 ms
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=t_cap, region=(0, 0, 4, 4))
    aff = tracker.ego.estimate(frame)

    yaw = math.radians(100.0 * 0.02)
    expected_tx = -focal_length_px(90.0, 1920) * math.tan(yaw)
    assert abs(aff[0, 2] - expected_tx) < 1e-3      # correct back-projected translation
    assert aff[0, 2] < 0.0                          # rightward pan -> world shifts left
    assert abs(aff[1, 2]) < 1e-9


def test_no_commanded_motion_is_identity_warp():
    buf = CommandedMotionBuffer()
    cfg = AppConfig(tracking={"gmc": "feedforward", "deg_per_count": 0.02})
    tracker = build_tracker(cfg, gmc_buffer=buf)
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=1_000_000_000, region=(0, 0, 4, 4))
    aff = tracker.ego.estimate(frame)               # empty buffer
    assert np.allclose(aff, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `python -m pytest tests/tracking/test_gmc_activation.py -v`
Expected: PASS (build_tracker from T3 wires the shared-buffer FeedForwardGMC; the warp math is the existing, already-tested FeedForwardGMC).

- [ ] **Step 3: Run the FULL suite**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 5B).

- [ ] **Step 4: Commit**

```bash
git add tests/tracking/test_gmc_activation.py
git commit -m "test(tracking): GMC activation integration (shared buffer -> non-identity warp)"
```

---

## Phase 5B completion checklist

- [ ] GMC active in the loop: tracker forwards the Frame (T1), config selects it (T2), wiring injects a shared-buffer FeedForwardGMC as the ego (T3) — **no vendored-core edit, no double-apply**, off by default.
- [ ] Calibration math: τ_render cross-correlation lag (T4) + signed deg_per_count (T5), pure + unit-tested.
- [ ] Activation proven end-to-end by a CI integration test (T6); identity default and all existing tests stay green.
- [ ] Scope-Boundary deferrals documented (live optical-flow τ_render capture + in-game wizard, passthrough deltas, recoil-into-ego-motion, world-angular target filter).

After merge: update memory (Phase 5B done; GMC live behind `tracking.gmc="feedforward"` + calibrated `deg_per_count`/`tau_render_s`). **Box-only smoke to enable it for real:** run the calibration turn to solve `deg_per_count` (sign + magnitude) and estimate `τ_render`, set them in config, then validate the residual collapses while panning across a static target (spec §18). Natural next: Plan 5C (dynamic-ROI) or Phase 6 (training pipeline).
