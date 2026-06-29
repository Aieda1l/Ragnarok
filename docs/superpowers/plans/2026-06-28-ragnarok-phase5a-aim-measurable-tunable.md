# Ragnarok Phase 5A — Aim Measurable & Tunable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the aim control loop **measurable** (a step-response diagnostics harness + a §15 CI regression) and then **tunable** (promote `FeedbackAimer` to a real 2-DOF PID, add relay-feedback + numeric auto-tune that emit gain *seeds*, and an explicit apply-seeds path).

**Architecture:** A new pure, CI-safe `src/ragnarok/diagnostics/` package: step-response metrics over a uniform `(t, y)` array, a jitter→uniform resampler (PCHIP, reusing the §6.6 recoil technique), a frozen result + recorder, a deterministic closed-loop plant simulator, and a transport-agnostic step-response runner (driven by injected `move`/`sample` callables so the same code serves CI fakes, the live desktop-cursor mode, and the in-game mode). On top of that: `FeedbackAimer` gains Ki/Kd + three-fold anti-windup (defaults `ki=kd=0` reproduce today's P behaviour exactly), and relay/numeric tuners run against the synthetic plant and return seeds that an explicit `apply_seeds` writes into a new frozen `AppConfig`.

**Tech Stack:** Python 3.11+, numpy, scipy (already a dep — `scipy.interpolate.PchipInterpolator`, `scipy.optimize.minimize`), pydantic/tomlkit config. No new third-party dependencies.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is control-systems / diagnostics engineering (spec §Scope, §11, §15).
- **CI-safe always:** no real GPU, display, cursor, or game in unit tests. Every live mode (desktop `GetCursorPos`, in-game detector-as-sensor, HIL) sits behind injected `move(dx,dy)` / `sample()->float` / `clock()` callables; CI wires them to a synthetic plant + a fake clock. Modules must import without torch/Win32/display.
- **Timing:** `time.perf_counter_ns()` (QPC), integer-ns math; convert to float seconds only inside the metric/resample layer. Do not do float arithmetic on raw ns (see `tracking/egomotion.py` which `round()`s ns to int).
- **Auto-tune emits SEEDS, not final values** (spec §11): a tuner never auto-writes config; `apply_seeds` is an explicit, separate call the caller wires to `ConfigHandle.swap`.
- **Backward compatibility:** the `FeedbackAimer` PID upgrade MUST default to today's P behaviour (`ki=0, kd=0`, no conditional integration) so existing `tests/aim/` and the new §15 regression stay green. Config additions are defaulted + frozen, TOML round-trip preserved.
- **No overshoot invariant** on the aimer's output stays in force (the `min(max_step_px, remaining_distance)` clamp).
- **Frozen pydantic config** (`model_config = ConfigDict(frozen=True)`); match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, module docstrings, focused files).
- **TDD, frequent commits, exact file paths.** Metrics/sim/tuner math is unit-tested against analytically-known responses; live capture is out of CI.

## Scope Boundary (explicit deferrals — Phase 5B / 5C / later)

- **Feed-forward GMC activation + τ_render/deg_per_count calibration** → **Plan 5B** (its own immediate follow-on). `FeedForwardGMC` is already built/tested in `tracking/egomotion.py` but unwired (worker still passes `IDENTITY_AFFINE` at `worker/loop.py`); 5B wires it, adds the cross-correlation τ_render solver (which reuses *this* plan's `resample.py`), and a signed `deg_per_count`. Out of 5A.
- **Dynamic-ROI (SEARCH/TRACK FSM, two-engine)** → **Plan 5C** (capture/detection throughput; shares no code with control diagnostics). Deferred.
- **HIL mode (c) / DIAG 0x04 echo** → deferred (needs the Phase-7 `firmware/` package). The runner's injected `sample()` seam can later accept an MCU-echo sampler.
- **GUI Diagnostics tab** → Phase 8. This plan ships the headless runner + frozen result objects + `apply_seeds`; those are the contract the later offscreen-Qt panel consumes.
- **CMA-ES tuner; plant-ID from logged step data** → deferred. `numeric_tune` abstracts the optimizer; scipy Nelder-Mead suffices and runs against the synthetic plant for CI.
- **Live modes are box-only smokes:** desktop `GetCursorPos` reader (mode a), in-game detector-as-sensor against a stationary dummy (mode b). Only the live capture is skipped; the analysis modules stay CI-tested via fakes.

---

## File Structure

**New files:**
- `src/ragnarok/diagnostics/__init__.py` — package marker.
- `src/ragnarok/diagnostics/metrics.py` — pure step-response metrics over uniform `(t, y)`.
- `src/ragnarok/diagnostics/resample.py` — jittery `(t_ns, value)` → uniform `(t_s, y)` (PCHIP).
- `src/ragnarok/diagnostics/results.py` — `StepResponseResult` (frozen) + `StepResponseRecorder` + `compute_step_result`.
- `src/ragnarok/diagnostics/plant.py` — `AimPlant` (integrator + optional lag/dead-time) + `simulate_closed_loop`.
- `src/ragnarok/diagnostics/runner.py` — `StepResponseRunner` (transport-agnostic live harness).
- `src/ragnarok/diagnostics/relay.py` — relay limit-cycle analysis + `ku_from_relay` + `zn_seed`.
- `src/ragnarok/diagnostics/relay_experiment.py` — `RelayController`, `RelayTuneResult`, `run_relay_tune`.
- `src/ragnarok/diagnostics/cost.py` — `itae_cost`.
- `src/ragnarok/diagnostics/numeric_tune.py` — `PidSeeds`, `numeric_tune`.
- `src/ragnarok/diagnostics/apply.py` — `apply_seeds`.
- `tests/diagnostics/__init__.py` + one test module per source module above.

**Modified files:**
- `src/ragnarok/aim/aimers.py` — `FeedbackAimer` → 2-DOF PID (Ki/Kd + anti-windup), backward-compatible defaults.
- `src/ragnarok/config/schema.py` — `DiagnosticsConfig` (+ nest in `AppConfig`); `AimConfig` gains `ki, kd, controller_mode, integral_clamp, cond_integ_thresh_px`.
- `src/ragnarok/wiring.py` — `build_aimer` threads the new PID gains (honoring `controller_mode`).

---

## Task 1: Step-response metrics (pure)

**Files:**
- Create: `src/ragnarok/diagnostics/__init__.py` (empty), `src/ragnarok/diagnostics/metrics.py`
- Create: `tests/diagnostics/__init__.py` (empty), `tests/diagnostics/test_metrics.py`

**Interfaces:**
- Produces (all operate on a uniform float `t` array in seconds + float `y` array, with the step's `y0` initial and `y_final` steady-state target):
  - `rise_time(t, y, *, y0, y_final, lo=0.1, hi=0.9) -> float | None`
  - `overshoot_pct(y, *, y0, y_final) -> float`
  - `settling_time(t, y, *, y0, y_final, band=0.02) -> float | None`
  - `dead_time(t, y, *, y0, y_final, dead_frac=0.05) -> float | None`
  - All normalize via `frac = (y - y0) / (y_final - y0)` (sign-agnostic). Degenerate `|y_final - y0| < 1e-9` → `rise/settling/dead = None`, `overshoot = 0.0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_metrics.py
"""Tests for pure step-response metrics against analytic responses."""
from __future__ import annotations
import math
import numpy as np
from ragnarok.diagnostics.metrics import rise_time, overshoot_pct, settling_time, dead_time


def _first_order(tau=0.1, dt=0.0005, t_end=2.0, K=1.0):
    t = np.arange(0.0, t_end, dt)
    y = K * (1.0 - np.exp(-t / tau))
    return t, y


def test_first_order_rise_time():
    t, y = _first_order(tau=0.1)
    r = rise_time(t, y, y0=0.0, y_final=1.0)
    assert abs(r - 0.1 * math.log(9.0)) < 5e-3   # tau*ln(9) ~= 0.2197 s


def test_first_order_no_overshoot():
    t, y = _first_order(tau=0.1)
    assert overshoot_pct(y, y0=0.0, y_final=1.0) < 0.5


def test_first_order_settling_2pct():
    t, y = _first_order(tau=0.1)
    s = settling_time(t, y, y0=0.0, y_final=1.0, band=0.02)
    assert abs(s - 0.1 * math.log(50.0)) < 1e-2   # tau*ln(50) ~= 0.3912 s


def test_second_order_overshoot_matches_zeta():
    # Underdamped 2nd-order step: overshoot% = 100*exp(-pi*z/sqrt(1-z^2))
    z, wn = 0.5, 30.0
    wd = wn * math.sqrt(1 - z * z)
    t = np.arange(0.0, 1.0, 0.0002)
    y = 1.0 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z * wn / wd) * np.sin(wd * t))
    expected = 100.0 * math.exp(-math.pi * z / math.sqrt(1 - z * z))   # ~16.3 %
    assert abs(overshoot_pct(y, y0=0.0, y_final=1.0) - expected) < 1.5


def test_never_settles_returns_none():
    t = np.arange(0.0, 1.0, 0.001)
    y = 1.0 + 0.5 * np.sin(50.0 * t)        # oscillates forever outside the band
    assert settling_time(t, y, y0=0.0, y_final=1.0, band=0.02) is None


def test_zero_step_is_guarded():
    t = np.arange(0.0, 1.0, 0.001)
    y = np.zeros_like(t)
    assert rise_time(t, y, y0=0.0, y_final=0.0) is None
    assert overshoot_pct(y, y0=0.0, y_final=0.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_metrics.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics'`.

- [ ] **Step 3: Implement metrics.py**

```python
# src/ragnarok/diagnostics/__init__.py
```

```python
# src/ragnarok/diagnostics/metrics.py
"""Pure step-response metrics (spec §11) over a UNIFORM (t, y) sample array.

Inputs are already resampled onto a uniform grid (see resample.py). All metrics
normalize the response to the commanded step via frac = (y - y0)/(y_final - y0)
so they are sign-agnostic. A degenerate step (|y_final - y0| < EPS) yields
None for time metrics and 0.0 overshoot.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _frac(y: np.ndarray, y0: float, y_final: float) -> np.ndarray | None:
    span = y_final - y0
    if abs(span) < _EPS:
        return None
    return (np.asarray(y, dtype=float) - y0) / span


def _first_cross_time(t: np.ndarray, frac: np.ndarray, level: float) -> float | None:
    """Linear-interpolated time at which frac first reaches `level`."""
    above = frac >= level
    if not above.any():
        return None
    i = int(np.argmax(above))            # first True
    if i == 0:
        return float(t[0])
    f0, f1 = frac[i - 1], frac[i]
    if f1 == f0:
        return float(t[i])
    w = (level - f0) / (f1 - f0)
    return float(t[i - 1] + w * (t[i] - t[i - 1]))


def rise_time(t, y, *, y0: float, y_final: float, lo: float = 0.1, hi: float = 0.9):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    t_lo = _first_cross_time(t, frac, lo)
    t_hi = _first_cross_time(t, frac, hi)
    if t_lo is None or t_hi is None:
        return None
    return t_hi - t_lo


def overshoot_pct(y, *, y0: float, y_final: float) -> float:
    frac = _frac(y, y0, y_final)
    if frac is None:
        return 0.0
    peak = float(np.max(frac))
    return max(0.0, (peak - 1.0) * 100.0)


def settling_time(t, y, *, y0: float, y_final: float, band: float = 0.02):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    outside = np.abs(frac - 1.0) > band
    if not outside.any():
        return 0.0
    last = int(np.max(np.flatnonzero(outside)))
    if last >= len(t) - 1:
        return None                       # still outside the band at the end
    return float(t[last + 1] - t[0])


def dead_time(t, y, *, y0: float, y_final: float, dead_frac: float = 0.05):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    tc = _first_cross_time(t, np.abs(frac), dead_frac)
    if tc is None:
        return None
    return tc - float(t[0])
```

```python
# tests/diagnostics/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/__init__.py src/ragnarok/diagnostics/metrics.py tests/diagnostics
git commit -m "feat(diagnostics): pure step-response metrics (rise/overshoot/settling/dead)"
```

---

## Task 2: Uniform-timeline resampler

**Files:**
- Create: `src/ragnarok/diagnostics/resample.py`
- Create: `tests/diagnostics/test_resample.py`

**Interfaces:**
- Consumes: nothing (scipy).
- Produces: `resample_uniform(t_ns, values, *, hz) -> tuple[np.ndarray, np.ndarray]` — returns `(t_s, y)` where `t_s` is a uniform grid in **seconds relative to the first sample** at `hz` samples/s spanning `[0, last-first]`, and `y` is the PCHIP-interpolated value at each grid point. Reuses the §6.6 monotone-PCHIP technique (no natural-cubic overshoot). Needs ≥2 samples; raises `ValueError` otherwise.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_resample.py
"""Tests for the jitter->uniform resampler."""
from __future__ import annotations
import numpy as np
import pytest
from ragnarok.diagnostics.resample import resample_uniform


def test_linear_ramp_is_preserved():
    # Jittered sample times; value is an exact linear ramp -> PCHIP reproduces it.
    t_ns = [0, 1_300_000, 2_900_000, 4_100_000, 5_000_000]   # ns, uneven
    vals = [0.0, 1.3, 2.9, 4.1, 5.0]                          # value == t in ms
    t_s, y = resample_uniform(t_ns, vals, hz=2000.0)
    assert t_s[0] == 0.0
    assert abs(t_s[-1] - 0.005) < 1e-9
    # at every uniform grid point, y(ms) ~= t_s*1000
    assert np.allclose(y, t_s * 1000.0, atol=1e-6)


def test_grid_spacing_matches_hz():
    t_ns = [0, 10_000_000]            # 10 ms span
    vals = [0.0, 1.0]
    t_s, y = resample_uniform(t_ns, vals, hz=1000.0)
    assert abs((t_s[1] - t_s[0]) - 0.001) < 1e-9   # 1 kHz -> 1 ms spacing


def test_requires_two_samples():
    with pytest.raises(ValueError):
        resample_uniform([5], [1.0], hz=1000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_resample.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.resample'`.

- [ ] **Step 3: Implement resample.py**

```python
# src/ragnarok/diagnostics/resample.py
"""Resample jittery (t_ns, value) samples onto a uniform timeline.

perf_counter_ns sample times are uneven and the mouse driver quantizes motion to
integer px, so step-response crossings/settling must be read off a uniform grid.
Uses monotone PCHIP (spec §6.6: PCHIP, not natural cubic which overshoots).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def resample_uniform(t_ns, values, *, hz: float) -> tuple[np.ndarray, np.ndarray]:
    t_ns = np.asarray(t_ns, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if t_ns.size < 2:
        raise ValueError("resample_uniform needs at least 2 samples")
    t_s = (t_ns - t_ns[0]) / 1e9                  # seconds relative to first sample
    span = float(t_s[-1])
    n = max(2, int(round(span * hz)) + 1)
    grid = np.linspace(0.0, span, n)
    pchip = PchipInterpolator(t_s, values, extrapolate=False)
    y = pchip(grid)
    return grid, y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_resample.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/resample.py tests/diagnostics/test_resample.py
git commit -m "feat(diagnostics): PCHIP jitter->uniform resampler"
```

---

## Task 3: StepResponseResult + recorder + compute helper

**Files:**
- Create: `src/ragnarok/diagnostics/results.py`
- Create: `tests/diagnostics/test_results.py`

**Interfaces:**
- Consumes: `resample_uniform` (T2), metrics (T1).
- Produces:
  - `@dataclass(frozen=True, eq=False) StepResponseResult` with `rise_s: float|None, overshoot_pct: float, settling_s: float|None, dead_time_s: float|None, t_s: np.ndarray, y: np.ndarray, y0: float, y_final: float`.
  - `StepResponseRecorder(*, clock=now_ns)`: `.record(value: float)` (stamps `clock()`), `.samples() -> tuple[np.ndarray, np.ndarray]` (t_ns int64, values float), `.reset()`.
  - `compute_step_result(t_ns, values, *, y0, y_final, hz, band=0.02, rise_lo=0.1, rise_hi=0.9, dead_frac=0.05) -> StepResponseResult` — resample then compute all metrics.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_results.py
"""Tests for StepResponseResult, recorder, and compute_step_result."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.results import (
    StepResponseRecorder, compute_step_result, StepResponseResult,
)


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def test_recorder_stamps_and_returns_samples():
    clk = _Clock()
    rec = StepResponseRecorder(clock=clk)
    clk.t = 1_000_000; rec.record(0.0)
    clk.t = 2_000_000; rec.record(1.0)
    t_ns, vals = rec.samples()
    assert list(t_ns) == [1_000_000, 2_000_000]
    assert list(vals) == [0.0, 1.0]
    rec.reset()
    assert rec.samples()[0].size == 0


def test_compute_step_result_on_first_order():
    tau, dt = 0.1, 0.0005
    t = np.arange(0.0, 2.0, dt)
    y = 1.0 - np.exp(-t / tau)
    t_ns = (t * 1e9).astype(np.int64)
    res = compute_step_result(t_ns, y, y0=0.0, y_final=1.0, hz=2000.0)
    assert isinstance(res, StepResponseResult)
    assert abs(res.rise_s - 0.2197) < 1e-2
    assert res.overshoot_pct < 0.5
    assert res.settling_s is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_results.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.results'`.

- [ ] **Step 3: Implement results.py**

```python
# src/ragnarok/diagnostics/results.py
"""Step-response result object, sample recorder, and the compute helper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragnarok.core.clock import now_ns
from ragnarok.diagnostics.resample import resample_uniform
from ragnarok.diagnostics import metrics


@dataclass(frozen=True, eq=False)
class StepResponseResult:
    rise_s: float | None
    overshoot_pct: float
    settling_s: float | None
    dead_time_s: float | None
    t_s: np.ndarray
    y: np.ndarray
    y0: float
    y_final: float


class StepResponseRecorder:
    """Accumulates (clock(), value) samples during a step-response run."""

    def __init__(self, *, clock=now_ns) -> None:
        self._clock = clock
        self._t: list[int] = []
        self._v: list[float] = []

    def record(self, value: float) -> None:
        self._t.append(int(self._clock()))
        self._v.append(float(value))

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self._t, dtype=np.int64), np.asarray(self._v, dtype=float))

    def reset(self) -> None:
        self._t.clear()
        self._v.clear()


def compute_step_result(
    t_ns, values, *, y0: float, y_final: float, hz: float,
    band: float = 0.02, rise_lo: float = 0.1, rise_hi: float = 0.9,
    dead_frac: float = 0.05,
) -> StepResponseResult:
    t_s, y = resample_uniform(t_ns, values, hz=hz)
    return StepResponseResult(
        rise_s=metrics.rise_time(t_s, y, y0=y0, y_final=y_final, lo=rise_lo, hi=rise_hi),
        overshoot_pct=metrics.overshoot_pct(y, y0=y0, y_final=y_final),
        settling_s=metrics.settling_time(t_s, y, y0=y0, y_final=y_final, band=band),
        dead_time_s=metrics.dead_time(t_s, y, y0=y0, y_final=y_final, dead_frac=dead_frac),
        t_s=t_s, y=y, y0=y0, y_final=y_final,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_results.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/results.py tests/diagnostics/test_results.py
git commit -m "feat(diagnostics): StepResponseResult, recorder, compute_step_result"
```

---

## Task 4: Aim plant + closed-loop simulator

**Files:**
- Create: `src/ragnarok/diagnostics/plant.py`
- Create: `tests/diagnostics/test_plant.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AimPlant(*, gain=1.0, lag_tau_s=0.0, dead_time_s=0.0, dt_s)` — discrete model of the aim plant (a mouse-delta command moves the crosshair, i.e. an **integrator**), with optional first-order actuator lag and pure dead-time. `.step(command: float) -> float` advances one `dt_s` and returns the new position. `.reset()`. `.position` property.
  - `simulate_closed_loop(controller_step, plant, *, setpoint, n_steps, dt_s, y0=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(t_s, measured, command)`. `controller_step` is a callable `(error: float, dt: float) -> command: float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_plant.py
"""Tests for the aim plant integrator + closed-loop simulator."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop


def test_integrator_accumulates_commands():
    p = AimPlant(dt_s=0.01)             # pure integrator, gain 1
    assert p.step(5.0) == 5.0
    assert p.step(3.0) == 8.0


def test_dead_time_delays_response():
    p = AimPlant(dt_s=0.01, dead_time_s=0.02)   # 2-tick delay
    assert p.step(10.0) == 0.0          # tick 1: still delayed
    assert p.step(0.0) == 0.0           # tick 2: still delayed
    assert p.step(0.0) == 10.0          # tick 3: the first command lands


def test_p_controller_on_integrator_is_first_order():
    # error -> command = kp*error; integrator closed loop: m_{k+1}=m_k+kp*(sp-m_k)
    # -> geometric approach to sp with ratio (1-kp).
    p = AimPlant(dt_s=0.01)
    kp = 0.3
    t, m, u = simulate_closed_loop(lambda e, dt: kp * e, p, setpoint=1.0,
                                   n_steps=200, dt_s=0.01)
    assert abs(m[-1] - 1.0) < 1e-3          # converges to setpoint
    assert m[0] == 0.3                      # first step = kp*1.0
    assert np.all(np.diff(m) >= -1e-9)      # monotone (no overshoot for P on integrator)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_plant.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.plant'`.

- [ ] **Step 3: Implement plant.py**

```python
# src/ragnarok/diagnostics/plant.py
"""Deterministic aim-plant model + closed-loop simulator (CI-safe, no IO).

The aim 'plant' is fundamentally an INTEGRATOR: each commanded mouse delta moves
the crosshair, so position += gain*command. Optional first-order actuator lag and
pure dead-time make it a more realistic FOPDT-ish plant for tuner coverage.
This lets the controller/auto-tuners be characterized entirely off-box.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AimPlant:
    def __init__(self, *, gain: float = 1.0, lag_tau_s: float = 0.0,
                 dead_time_s: float = 0.0, dt_s: float) -> None:
        self._gain = gain
        self._dt = dt_s
        self._lag_tau = lag_tau_s
        self._pos = 0.0
        self._lagged = 0.0                       # actuator-lag state
        delay = max(0, int(round(dead_time_s / dt_s)))
        self._delay = deque([0.0] * delay, maxlen=delay) if delay else None

    @property
    def position(self) -> float:
        return self._pos

    def reset(self) -> None:
        self._pos = 0.0
        self._lagged = 0.0
        if self._delay is not None:
            self._delay = deque([0.0] * self._delay.maxlen, maxlen=self._delay.maxlen)

    def step(self, command: float) -> float:
        u = command
        if self._delay is not None:              # pure dead-time
            self._delay.append(u)
            u = self._delay[0]
        if self._lag_tau > 0.0:                  # first-order actuator lag
            a = self._dt / (self._lag_tau + self._dt)
            self._lagged += a * (u - self._lagged)
            u = self._lagged
        self._pos += self._gain * u              # integrator
        return self._pos


def simulate_closed_loop(controller_step, plant: AimPlant, *, setpoint: float,
                         n_steps: int, dt_s: float, y0: float = 0.0):
    plant.reset()
    measured = y0
    t_s = np.empty(n_steps)
    m_arr = np.empty(n_steps)
    u_arr = np.empty(n_steps)
    for k in range(n_steps):
        error = setpoint - measured
        command = controller_step(error, dt_s)
        measured = y0 + plant.step(command)
        t_s[k] = k * dt_s
        m_arr[k] = measured
        u_arr[k] = command
    return t_s, m_arr, u_arr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_plant.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/plant.py tests/diagnostics/test_plant.py
git commit -m "feat(diagnostics): AimPlant integrator + closed-loop simulator"
```

---

## Task 5: Step-response runner (transport-agnostic live harness)

**Files:**
- Create: `src/ragnarok/diagnostics/runner.py`
- Create: `tests/diagnostics/test_runner.py`

**Interfaces:**
- Consumes: `StepResponseRecorder`, `compute_step_result` (T3).
- Produces: `StepResponseRunner(*, move, sample, clock=now_ns, step_px, hz, timeout_s, axis="x", band=0.02, rise_lo=0.1, rise_hi=0.9, dead_frac=0.05)`. `move(dx, dy)` injects motion; `sample() -> float` reads the position on the chosen axis. `.run() -> StepResponseResult`: read `y0 = sample()`, inject the step once via `move(step_px, 0)` (or `(0, step_px)` for axis "y"), then poll-and-record `sample()` each tick until `timeout_s` of `clock()` elapses, then `compute_step_result` against `y_final = y0 + step_px`. CI drives `move`/`sample` against an `AimPlant`; the live desktop/in-game modes pass real callables (box-only).

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_runner.py
"""Tests for the step-response runner against a synthetic plant (no real IO)."""
from __future__ import annotations
from ragnarok.diagnostics.runner import StepResponseRunner
from ragnarok.diagnostics.plant import AimPlant


class _Clock:
    def __init__(self, dt_ns):
        self.t = 0
        self._dt = dt_ns
    def __call__(self):
        v = self.t
        self.t += self._dt          # each read advances time by one tick
        return v


def test_runner_characterizes_first_order_plant():
    # A first-order actuator-lag plant fed a single step is a first-order response.
    dt = 0.001
    plant = AimPlant(dt_s=dt, lag_tau_s=0.05)
    pos = {"v": 0.0}

    def move(dx, dy):
        # the runner injects the step once; the plant integrates it over time
        move.cmd = dx
    move.cmd = 0.0

    def sample():
        pos["v"] = plant.step(move.cmd)
        move.cmd = 0.0              # the step is a one-shot impulse of size step_px
        return pos["v"]

    clk = _Clock(int(dt * 1e9))
    runner = StepResponseRunner(move=move, sample=sample, clock=clk,
                                step_px=10.0, hz=1000.0, timeout_s=1.0)
    res = runner.run()
    assert res.y0 == 0.0
    assert res.y_final == 10.0
    assert res.rise_s is not None and res.rise_s > 0.0
    assert res.overshoot_pct < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_runner.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.runner'`.

- [ ] **Step 3: Implement runner.py**

```python
# src/ragnarok/diagnostics/runner.py
"""Transport-agnostic step-response runner (spec §11 modes a/b).

Inject a known step via move(dx,dy), poll sample()->position until timeout, then
compute the step-response metrics. The move/sample/clock seams keep it CI-safe:
tests drive a synthetic AimPlant; live desktop (GetCursorPos) and in-game
(detector-as-sensor) modes pass real callables (box-only smokes).
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.diagnostics.results import StepResponseRecorder, compute_step_result


class StepResponseRunner:
    def __init__(self, *, move, sample, clock=now_ns, step_px: float, hz: float,
                 timeout_s: float, axis: str = "x", band: float = 0.02,
                 rise_lo: float = 0.1, rise_hi: float = 0.9, dead_frac: float = 0.05) -> None:
        self._move = move
        self._sample = sample
        self._clock = clock
        self._step = step_px
        self._hz = hz
        self._timeout_ns = int(timeout_s * 1e9)
        self._axis = axis
        self._band = band
        self._rise_lo = rise_lo
        self._rise_hi = rise_hi
        self._dead_frac = dead_frac

    def run(self):
        rec = StepResponseRecorder(clock=self._clock)
        y0 = float(self._sample())
        if self._axis == "y":
            self._move(0.0, self._step)
        else:
            self._move(self._step, 0.0)
        t_start = self._clock()
        while True:
            rec.record(self._sample())
            if self._clock() - t_start >= self._timeout_ns:
                break
        t_ns, vals = rec.samples()
        return compute_step_result(
            t_ns, vals, y0=y0, y_final=y0 + self._step, hz=self._hz,
            band=self._band, rise_lo=self._rise_lo, rise_hi=self._rise_hi,
            dead_frac=self._dead_frac,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/runner.py tests/diagnostics/test_runner.py
git commit -m "feat(diagnostics): transport-agnostic step-response runner"
```

---

## Task 6: DiagnosticsConfig

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Create: `tests/config/test_diagnostics_config.py`

**Interfaces:**
- Produces: `DiagnosticsConfig` (frozen) with `step_px=200.0 (gt 0)`, `sample_hz=1000.0 (gt 0)`, `timeout_s=1.0 (gt 0, le 30)`, `settle_band_frac=0.02 (gt 0, lt 1)`, `rise_lo=0.1 (gt 0, lt 1)`, `rise_hi=0.9 (gt rise_lo? — keep gt 0, lt 1)`, `dead_frac=0.05 (ge 0, lt 1)`, `reg_max_overshoot_pct=5.0 (ge 0)`. Nested as `AppConfig.diagnostics`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_diagnostics_config.py
"""Tests for DiagnosticsConfig + its nesting in AppConfig."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import DiagnosticsConfig, AppConfig


def test_defaults():
    d = DiagnosticsConfig()
    assert d.step_px == 200.0
    assert d.sample_hz == 1000.0
    assert d.timeout_s == 1.0
    assert d.settle_band_frac == 0.02
    assert d.rise_lo == 0.1 and d.rise_hi == 0.9
    assert d.dead_frac == 0.05
    assert d.reg_max_overshoot_pct == 5.0


def test_bounds():
    with pytest.raises(ValidationError):
        DiagnosticsConfig(step_px=0.0)
    with pytest.raises(ValidationError):
        DiagnosticsConfig(settle_band_frac=1.0)


def test_nested_and_backward_compatible():
    assert isinstance(AppConfig().diagnostics, DiagnosticsConfig)
    app = AppConfig(detection={"model": "nano"})
    assert app.diagnostics.sample_hz == 1000.0


def test_frozen():
    with pytest.raises(Exception):
        DiagnosticsConfig().step_px = 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_diagnostics_config.py -v`
Expected: FAIL — `cannot import name 'DiagnosticsConfig'`.

- [ ] **Step 3: Add DiagnosticsConfig and nest it**

In `src/ragnarok/config/schema.py`, add (before `AppConfig`):

```python
class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    step_px: float = Field(default=200.0, gt=0.0)
    sample_hz: float = Field(default=1000.0, gt=0.0)
    timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    settle_band_frac: float = Field(default=0.02, gt=0.0, lt=1.0)
    rise_lo: float = Field(default=0.1, gt=0.0, lt=1.0)
    rise_hi: float = Field(default=0.9, gt=0.0, lt=1.0)
    dead_frac: float = Field(default=0.05, ge=0.0, lt=1.0)
    reg_max_overshoot_pct: float = Field(default=5.0, ge=0.0)   # §15 regression bound
```

Add to `AppConfig` (after `trigger`):

```python
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config -q`
Expected: PASS (incl. the TOML round-trip in `test_store.py`).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py tests/config/test_diagnostics_config.py
git commit -m "feat(config): DiagnosticsConfig (step-response harness params + §15 bound)"
```

---

## Task 7: §15 CI step-response regression (locks current P behaviour)

**Files:**
- Create: `tests/diagnostics/test_regression.py`

**Interfaces:**
- Consumes: `FeedbackAimer` (current), `simulate_closed_loop` + `AimPlant` (T4), `compute_step_result` (T3), `DiagnosticsConfig` (T6).
- Produces: a CI regression test that drives the **current** `FeedbackAimer` (P-controller) closed-loop against `AimPlant` and asserts the response is well-behaved (overshoot under the config bound, settles, monotone-ish). This locks today's behaviour **before** the PID upgrade (Task 8), guarding the §6.4 stability claim.

- [ ] **Step 1: Write the test (this is the deliverable)**

```python
# tests/diagnostics/test_regression.py
"""§15 step-response regression: the closed loop must stay well-behaved.

Runs BEFORE the Task 8 PID change to lock current P behaviour, and stays green
after (ki=kd=0 defaults reproduce P). Drives FeedbackAimer against the synthetic
integrator plant — fully CI-safe (no GPU/cursor/game).
"""
from __future__ import annotations
import numpy as np
from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.config.schema import DiagnosticsConfig
from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop
from ragnarok.diagnostics.results import compute_step_result


def _drive(aimer, *, setpoint, dt, n):
    aimer.reset()
    plant = AimPlant(dt_s=dt)

    def ctrl(error, dt_):
        return aimer.step((0.0, 0.0), (error, 0.0), dt_)[0]   # 1-D: x-axis only

    t_s, measured, _ = simulate_closed_loop(ctrl, plant, setpoint=setpoint,
                                            n_steps=n, dt_s=dt)
    t_ns = (t_s * 1e9).astype(np.int64)
    return compute_step_result(t_ns, measured, y0=0.0, y_final=setpoint, hz=1.0 / dt)


def test_feedback_p_controller_is_well_behaved():
    cfg = DiagnosticsConfig()
    # Pure P (ema_alpha=1, kff=0), large max_step so the clamp doesn't slew-limit.
    aimer = FeedbackAimer(kp=0.3, max_step_px=1e9, ema_alpha=1.0)
    res = _drive(aimer, setpoint=100.0, dt=0.002, n=600)
    assert res.overshoot_pct <= cfg.reg_max_overshoot_pct   # P on integrator: ~0
    assert res.settling_s is not None                       # it settles
    assert res.rise_s is not None and res.rise_s > 0.0
```

- [ ] **Step 2: Run the test to verify it passes against current code**

Run: `python -m pytest tests/diagnostics/test_regression.py -v`
Expected: PASS (the current P-controller on an integrator has ~0 overshoot and settles). If it FAILS, stop — that is a real signal about the current controller, not a test bug.

- [ ] **Step 3: Commit**

```bash
git add tests/diagnostics/test_regression.py
git commit -m "test(diagnostics): §15 closed-loop step-response regression (locks P behaviour)"
```

---

## Task 8: Promote FeedbackAimer to 2-DOF PID (backward-compatible)

**Files:**
- Modify: `src/ragnarok/aim/aimers.py`
- Test: `tests/aim/test_feedback_pid.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `FeedbackAimer(*, kp, max_step_px, ema_alpha=1.0, kff=0.0, ki=0.0, kd=0.0, integral_clamp=None, cond_integ_thresh_px=None)`. Adds: per-axis integral with **conditional integration** (only when `|error| <= cond_integ_thresh_px`, or always if `None`), an **integral-contribution clamp** (`±integral_clamp` on `ki*integral`, if set), **derivative on the EMA-filtered error**, and **freeze-on-saturation anti-windup** (back out the integral increment when the magnitude clamp fires). `reset()` clears the EMA seed AND the integral. Defaults (`ki=0, kd=0, clamp=None, cond=None`) reproduce today's P path exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_feedback_pid.py
"""Tests for FeedbackAimer's 2-DOF PID upgrade (Ki/Kd + anti-windup)."""
from __future__ import annotations
from ragnarok.aim.aimers import FeedbackAimer


def test_defaults_reproduce_p_controller():
    a = FeedbackAimer(kp=0.5, max_step_px=1e9, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), 0.01)
    assert abs(dx - 5.0) < 1e-9 and abs(dy) < 1e-9     # pure P, unchanged


def test_integral_accumulates_and_adds():
    a = FeedbackAimer(kp=0.0, ki=2.0, max_step_px=1e9, ema_alpha=1.0)
    # kp=0 isolates I. error=10 held; integral grows by error*dt each step.
    a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral=1.0 -> ki*I=2.0
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral=2.0 -> ki*I=4.0
    assert abs(dx - 4.0) < 1e-9


def test_integral_contribution_clamp():
    a = FeedbackAimer(kp=0.0, ki=10.0, integral_clamp=3.0, max_step_px=1e9, ema_alpha=1.0)
    for _ in range(10):
        dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)
    assert abs(dx - 3.0) < 1e-9          # ki*I clamped to +3


def test_conditional_integration_only_when_close():
    a = FeedbackAimer(kp=0.0, ki=5.0, cond_integ_thresh_px=5.0,
                      max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (100.0, 0.0), 0.1)   # |e|=100 > 5 -> no integration
    dx, _ = a.step((0.0, 0.0), (100.0, 0.0), 0.1)
    assert abs(dx) < 1e-9                    # integral still 0


def test_derivative_on_filtered_error_opposes_rapid_approach():
    a = FeedbackAimer(kp=0.0, kd=1.0, max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (0.0, 0.0), 0.1)     # error 0 (seed)
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # error jumps 0->10, deriv=+100
    assert dx > 0.0                          # kd*derivative term present


def test_reset_clears_integral():
    a = FeedbackAimer(kp=0.0, ki=2.0, max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (10.0, 0.0), 0.1)
    a.reset()
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral restarts at error*dt
    assert abs(dx - 2.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_feedback_pid.py -v`
Expected: FAIL — `FeedbackAimer.__init__() got an unexpected keyword argument 'ki'`.

- [ ] **Step 3: Implement the PID upgrade**

Replace the `FeedbackAimer` class body in `src/ragnarok/aim/aimers.py` with:

```python
class FeedbackAimer(Aimer):
    """2-DOF PID: u = Kp·ē + Ki·∫e + Kd·d(ē)/dt + Kff·v̂ (spec §6.3).

    ē is the EMA-filtered error; the derivative is taken on the FILTERED error
    (no derivative kick). Three-fold anti-windup (spec §6.3): conditional
    integration (only when |e| <= cond_integ_thresh_px), an integral-contribution
    clamp (±integral_clamp on Ki·∫e), and freeze-on-saturation (back out the
    integral increment when the magnitude clamp fires). Output is magnitude-
    clamped to min(max_step_px, remaining distance) — never overshoots.

    Defaults (ki=0, kd=0, integral_clamp=None, cond_integ_thresh_px=None)
    reproduce the original P-controller behaviour exactly.
    """

    def __init__(self, *, kp: float, max_step_px: float, ema_alpha: float = 1.0,
                 kff: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 integral_clamp: float | None = None,
                 cond_integ_thresh_px: float | None = None) -> None:
        self._kp = kp
        self._max = max_step_px
        self._alpha = ema_alpha
        self._kff = kff
        self._ki = ki
        self._kd = kd
        self._iclamp = integral_clamp
        self._cond = cond_integ_thresh_px
        self._fx = 0.0
        self._fy = 0.0
        self._ix = 0.0
        self._iy = 0.0
        self._prev_fx = 0.0
        self._prev_fy = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        self._ix = 0.0
        self._iy = 0.0

    def step(self, crosshair, target_point, dt, target_vel=(0.0, 0.0)):
        ex = target_point[0] - crosshair[0]
        ey = target_point[1] - crosshair[1]

        if not self._initialized:
            self._fx, self._fy = ex, ey
            self._prev_fx, self._prev_fy = ex, ey
            self._initialized = True
            dfx = dfy = 0.0
        else:
            a = self._alpha
            self._fx += a * (ex - self._fx)
            self._fy += a * (ey - self._fy)
            if dt > 0.0:
                dfx = (self._fx - self._prev_fx) / dt
                dfy = (self._fy - self._prev_fy) / dt
            else:
                dfx = dfy = 0.0
            self._prev_fx, self._prev_fy = self._fx, self._fy

        # Conditional integration (anti-windup #1).
        e_mag = math.hypot(ex, ey)
        integrate = self._cond is None or e_mag <= self._cond
        inc_x = ex * dt if integrate else 0.0
        inc_y = ey * dt if integrate else 0.0
        self._ix += inc_x
        self._iy += inc_y

        # Integral contribution, clamped (anti-windup #2).
        icx = self._ki * self._ix
        icy = self._ki * self._iy
        if self._iclamp is not None:
            icx = max(-self._iclamp, min(self._iclamp, icx))
            icy = max(-self._iclamp, min(self._iclamp, icy))

        dx = self._kp * self._fx + icx + self._kd * dfx + self._kff * target_vel[0] * dt
        dy = self._kp * self._fy + icy + self._kd * dfy + self._kff * target_vel[1] * dt

        # Magnitude clamp: never overshoot remaining distance OR max step.
        mag = math.hypot(dx, dy)
        limit = min(self._max, e_mag)
        if mag > limit and mag > 0.0:
            scale = limit / mag
            dx *= scale
            dy *= scale
            # Freeze-on-saturation (anti-windup #3): undo this step's integration.
            self._ix -= inc_x
            self._iy -= inc_y

        return (dx, dy)
```

- [ ] **Step 4: Run the new tests + the existing aim suite + the §15 regression**

Run: `python -m pytest tests/aim tests/diagnostics/test_regression.py -q`
Expected: PASS (new PID tests, all existing `tests/aim/` including `test_aimers.py`/`test_controller*.py`, and the Task-7 regression — `ki=kd=0` defaults keep them identical).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/aimers.py tests/aim/test_feedback_pid.py
git commit -m "feat(aim): FeedbackAimer 2-DOF PID with three-fold anti-windup (P-compatible defaults)"
```

---

## Task 9: AimConfig PID fields + wiring

**Files:**
- Modify: `src/ragnarok/config/schema.py`, `src/ragnarok/wiring.py`
- Test: `tests/config/test_phase4_config.py` (extend) and `tests/test_wiring.py` (extend)

**Interfaces:**
- Consumes: `FeedbackAimer` PID params (T8).
- Produces: `AimConfig` gains `ki: float = 0.0 (ge 0)`, `kd: float = 0.0 (ge 0)`, `controller_mode: Literal["p","pi","pid"] = "p"`, `integral_clamp: float | None = None (gt 0 when set)`, `cond_integ_thresh_px: float | None = None (gt 0 when set)`. `build_aimer` for the `"feedback"` aimer applies `controller_mode`: `"p"` → `ki=kd=0`; `"pi"` → `kd=0`; `"pid"` → both — passing `integral_clamp`/`cond_integ_thresh_px` through.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/config/test_phase4_config.py
def test_aim_pid_defaults():
    from ragnarok.config.schema import AimConfig
    a = AimConfig()
    assert a.ki == 0.0 and a.kd == 0.0
    assert a.controller_mode == "p"
    assert a.integral_clamp is None and a.cond_integ_thresh_px is None


def test_controller_mode_validated():
    import pytest
    from pydantic import ValidationError
    from ragnarok.config.schema import AimConfig
    with pytest.raises(ValidationError):
        AimConfig(controller_mode="pdf")
```

```python
# append to tests/test_wiring.py
def test_build_aimer_feedback_mode_p_zeroes_gains():
    from ragnarok.wiring import build_aimer
    a = build_aimer(AppConfig(aim={"aimer": "feedback", "controller_mode": "p",
                                   "ki": 9.0, "kd": 9.0}))
    assert a._ki == 0.0 and a._kd == 0.0     # mode 'p' overrides the gains


def test_build_aimer_feedback_mode_pid_applies_gains():
    from ragnarok.wiring import build_aimer
    a = build_aimer(AppConfig(aim={"aimer": "feedback", "controller_mode": "pid",
                                   "ki": 0.2, "kd": 0.05}))
    assert a._ki == 0.2 and a._kd == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_phase4_config.py -k pid tests/test_wiring.py -k mode -v`
Expected: FAIL — `AimConfig` has no `controller_mode` / `build_aimer` ignores it.

- [ ] **Step 3: Add the fields and wire them**

In `src/ragnarok/config/schema.py`, add to `AimConfig` (after `lead_alpha`):

```python
    # --- Phase 5A PID additions ---
    ki: float = Field(default=0.0, ge=0.0)
    kd: float = Field(default=0.0, ge=0.0)
    controller_mode: Literal["p", "pi", "pid"] = "p"
    integral_clamp: float | None = Field(default=None, gt=0.0)
    cond_integ_thresh_px: float | None = Field(default=None, gt=0.0)
```

In `src/ragnarok/wiring.py`, replace the `FeedbackAimer` construction in `build_aimer` (the final `return FeedbackAimer(...)`) with:

```python
    ki = a.ki if a.controller_mode in ("pi", "pid") else 0.0
    kd = a.kd if a.controller_mode == "pid" else 0.0
    return FeedbackAimer(
        kp=a.kp, max_step_px=a.max_step_px, ema_alpha=a.ema_alpha, kff=a.kff,
        ki=ki, kd=kd, integral_clamp=a.integral_clamp,
        cond_integ_thresh_px=a.cond_integ_thresh_px,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config tests/test_wiring.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py src/ragnarok/wiring.py tests/config/test_phase4_config.py tests/test_wiring.py
git commit -m "feat(config,wiring): PID gains + controller_mode for FeedbackAimer"
```

---

## Task 10: Relay limit-cycle analysis + Ziegler-Nichols seeding (pure)

**Files:**
- Create: `src/ragnarok/diagnostics/relay.py`
- Create: `tests/diagnostics/test_relay.py`

**Interfaces:**
- Consumes: nothing (numpy).
- Produces:
  - `limit_cycle(t, y) -> tuple[float, float]` — `(amplitude_a, period_Tu)` from a steady oscillation: amplitude = `(max-min)/2` over the last portion; period from successive same-direction peak spacings (median).
  - `ku_from_relay(d, a) -> float` — `4*d/(π*a)` (relay amplitude `d`, limit-cycle amplitude `a`).
  - `zn_seed(Ku, Tu, *, rule="low_overshoot") -> dict[str,float]` — returns `{"kp","ki","kd"}`. `"classic"`: `Kp=0.6Ku, Ki=1.2Ku/Tu, Kd=0.075Ku·Tu`. `"pi"`: `Kp=0.45Ku, Ki=0.54Ku/Tu, Kd=0`. `"low_overshoot"` (default, precision aim — Pessen/no-overshoot biased): `Kp=0.2Ku, Ki=0.4Ku/Tu, Kd=0.066Ku·Tu`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_relay.py
"""Tests for relay limit-cycle analysis + ZN seeding (pure)."""
from __future__ import annotations
import math
import numpy as np
from ragnarok.diagnostics.relay import limit_cycle, ku_from_relay, zn_seed


def test_limit_cycle_amplitude_and_period():
    T = 0.2
    t = np.arange(0.0, 2.0, 0.001)
    y = 1.0 + 3.0 * np.sin(2 * math.pi * t / T)   # amplitude 3, period 0.2
    a, Tu = limit_cycle(t, y)
    assert abs(a - 3.0) < 0.1
    assert abs(Tu - 0.2) < 0.01


def test_ku_from_relay_formula():
    assert abs(ku_from_relay(d=1.0, a=2.0) - (4.0 / (math.pi * 2.0))) < 1e-9


def test_zn_seed_classic():
    s = zn_seed(Ku=10.0, Tu=0.5, rule="classic")
    assert abs(s["kp"] - 6.0) < 1e-9            # 0.6*Ku
    assert abs(s["ki"] - 24.0) < 1e-9           # 1.2*Ku/Tu
    assert abs(s["kd"] - 0.375) < 1e-9          # 0.075*Ku*Tu


def test_zn_seed_low_overshoot_is_gentler_than_classic():
    classic = zn_seed(Ku=10.0, Tu=0.5, rule="classic")
    low = zn_seed(Ku=10.0, Tu=0.5, rule="low_overshoot")
    assert low["kp"] < classic["kp"]            # precision loop -> less aggressive
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_relay.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.relay'`.

- [ ] **Step 3: Implement relay.py**

```python
# src/ragnarok/diagnostics/relay.py
"""Relay-feedback (Åström-Hägglund) limit-cycle analysis + ZN seeding (spec §11).

ku_from_relay: Ku = 4d/(π·a) from relay amplitude d and limit-cycle amplitude a.
zn_seed: Ziegler-Nichols gains; default 'low_overshoot' (Pessen-style) for the
precision-aim loop. These are SEEDS for tuning, never final values (spec §11).
"""
from __future__ import annotations

import math

import numpy as np


def _peaks(y: np.ndarray) -> np.ndarray:
    """Indices of local maxima (strict interior)."""
    return np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1


def limit_cycle(t, y) -> tuple[float, float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    half = len(y) // 2                       # analyze the steady tail
    tail_y = y[half:]
    tail_t = t[half:]
    amplitude = (float(np.max(tail_y)) - float(np.min(tail_y))) / 2.0
    pk = _peaks(tail_y)
    if pk.size >= 2:
        period = float(np.median(np.diff(tail_t[pk])))
    else:
        period = 0.0
    return amplitude, period


def ku_from_relay(d: float, a: float) -> float:
    return 4.0 * d / (math.pi * a)


def zn_seed(Ku: float, Tu: float, *, rule: str = "low_overshoot") -> dict[str, float]:
    if rule == "classic":
        return {"kp": 0.6 * Ku, "ki": 1.2 * Ku / Tu, "kd": 0.075 * Ku * Tu}
    if rule == "pi":
        return {"kp": 0.45 * Ku, "ki": 0.54 * Ku / Tu, "kd": 0.0}
    if rule == "low_overshoot":
        return {"kp": 0.2 * Ku, "ki": 0.4 * Ku / Tu, "kd": 0.066 * Ku * Tu}
    raise ValueError(f"unknown ZN rule {rule!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_relay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/relay.py tests/diagnostics/test_relay.py
git commit -m "feat(diagnostics): relay limit-cycle analysis + Ziegler-Nichols seeding"
```

---

## Task 11: Relay experiment against the plant

**Files:**
- Create: `src/ragnarok/diagnostics/relay_experiment.py`
- Create: `tests/diagnostics/test_relay_experiment.py`

**Interfaces:**
- Consumes: `limit_cycle`, `ku_from_relay`, `zn_seed` (T10), `AimPlant`, `simulate_closed_loop` (T4).
- Produces:
  - `RelayController(*, d, hysteresis=0.0)` — `(error, dt) -> ±d` with hysteresis (bang-bang).
  - `@dataclass(frozen=True) RelayTuneResult(ku, tu, kp, ki, kd)`.
  - `run_relay_tune(plant, *, d, n_steps, dt_s, setpoint=0.0, hysteresis=0.0, rule="low_overshoot") -> RelayTuneResult` — drives the relay closed-loop against `plant`, analyzes the limit cycle, returns Ku/Tu + seeded gains.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_relay_experiment.py
"""Tests for the relay experiment against the synthetic plant."""
from __future__ import annotations
from ragnarok.diagnostics.relay_experiment import RelayController, run_relay_tune, RelayTuneResult
from ragnarok.diagnostics.plant import AimPlant


def test_relay_controller_bangs():
    r = RelayController(d=2.0)
    assert r.step(5.0, 0.01) == 2.0       # positive error -> +d
    assert r.step(-5.0, 0.01) == -2.0     # negative error -> -d


def test_relay_tune_yields_positive_gains_on_lagged_plant():
    # An integrator with dead-time + lag sustains a limit cycle under relay.
    plant = AimPlant(dt_s=0.001, lag_tau_s=0.02, dead_time_s=0.01)
    res = run_relay_tune(plant, d=5.0, n_steps=4000, dt_s=0.001)
    assert isinstance(res, RelayTuneResult)
    assert res.tu > 0.0 and res.ku > 0.0
    assert res.kp > 0.0 and res.ki > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_relay_experiment.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.relay_experiment'`.

- [ ] **Step 3: Implement relay_experiment.py**

```python
# src/ragnarok/diagnostics/relay_experiment.py
"""Run a relay-feedback experiment against a plant to seed PID gains (spec §11).

CI runs this against the synthetic AimPlant; live desktop/in-game samplers are
thin box-only adapters over the same (error -> command) seam.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop
from ragnarok.diagnostics.relay import limit_cycle, ku_from_relay, zn_seed


class RelayController:
    def __init__(self, *, d: float, hysteresis: float = 0.0) -> None:
        self._d = d
        self._h = hysteresis
        self._out = d

    def step(self, error: float, dt: float) -> float:
        if error > self._h:
            self._out = self._d
        elif error < -self._h:
            self._out = -self._d
        # within the hysteresis band: hold previous output
        return self._out


@dataclass(frozen=True)
class RelayTuneResult:
    ku: float
    tu: float
    kp: float
    ki: float
    kd: float


def run_relay_tune(plant: AimPlant, *, d: float, n_steps: int, dt_s: float,
                   setpoint: float = 0.0, hysteresis: float = 0.0,
                   rule: str = "low_overshoot") -> RelayTuneResult:
    relay = RelayController(d=d, hysteresis=hysteresis)
    t_s, measured, _ = simulate_closed_loop(relay.step, plant, setpoint=setpoint,
                                            n_steps=n_steps, dt_s=dt_s)
    a, tu = limit_cycle(t_s, measured)
    ku = ku_from_relay(d, a) if a > 0.0 else 0.0
    seed = zn_seed(ku, tu, rule=rule) if (ku > 0.0 and tu > 0.0) else {"kp": 0.0, "ki": 0.0, "kd": 0.0}
    return RelayTuneResult(ku=ku, tu=tu, kp=seed["kp"], ki=seed["ki"], kd=seed["kd"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_relay_experiment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/relay_experiment.py tests/diagnostics/test_relay_experiment.py
git commit -m "feat(diagnostics): relay experiment -> Ku/Tu/PID seed against the plant"
```

---

## Task 12: ITAE cost + numeric (Nelder-Mead) tuner

**Files:**
- Create: `src/ragnarok/diagnostics/cost.py`, `src/ragnarok/diagnostics/numeric_tune.py`
- Create: `tests/diagnostics/test_numeric_tune.py`

**Interfaces:**
- Consumes: `simulate_closed_loop`, `AimPlant` (T4), `FeedbackAimer` (T8), `overshoot_pct` (T1).
- Produces:
  - `itae_cost(t, measured, command, *, setpoint, y0=0.0, w_overshoot=1.0, w_effort=0.0) -> float` — `Σ t·|error|·dt + w_overshoot·overshoot% + w_effort·Σ|command|`.
  - `@dataclass(frozen=True) PidSeeds(kp, ki, kd)`.
  - `numeric_tune(plant_factory, *, seed: PidSeeds, setpoint, n_steps, dt_s, max_step_px=1e9, w_overshoot=1.0, w_effort=0.0) -> PidSeeds` — `scipy.optimize.minimize(method="Nelder-Mead")` over `(kp, ki, kd)` (clamped ≥0), each evaluation building a fresh `FeedbackAimer(controller… pid)` + a fresh plant via `plant_factory()` and scoring with `itae_cost`. Deterministic (no randomness in the loop).

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_numeric_tune.py
"""Tests for the ITAE cost and the Nelder-Mead numeric tuner."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.cost import itae_cost
from ragnarok.diagnostics.numeric_tune import numeric_tune, PidSeeds
from ragnarok.diagnostics.plant import AimPlant


def test_itae_penalizes_slow_response():
    t = np.linspace(0, 1, 100)
    fast = np.ones_like(t)               # at setpoint immediately
    slow = np.linspace(0, 1, 100)        # ramps up slowly
    cmd = np.zeros_like(t)
    c_fast = itae_cost(t, fast, cmd, setpoint=1.0)
    c_slow = itae_cost(t, slow, cmd, setpoint=1.0)
    assert c_fast < c_slow


def test_numeric_tune_lowers_cost_vs_seed():
    def plant_factory():
        return AimPlant(dt_s=0.005, lag_tau_s=0.03, dead_time_s=0.01)

    seed = PidSeeds(kp=0.05, ki=0.0, kd=0.0)
    tuned = numeric_tune(plant_factory, seed=seed, setpoint=100.0,
                         n_steps=400, dt_s=0.005)
    assert isinstance(tuned, PidSeeds)

    from ragnarok.diagnostics.plant import simulate_closed_loop
    from ragnarok.aim.aimers import FeedbackAimer

    def score(s):
        a = FeedbackAimer(kp=s.kp, ki=s.ki, kd=s.kd, max_step_px=1e9, ema_alpha=1.0)
        t, m, u = simulate_closed_loop(lambda e, dt: a.step((0, 0), (e, 0), dt)[0],
                                       plant_factory(), setpoint=100.0,
                                       n_steps=400, dt_s=0.005)
        return itae_cost(t, m, u, setpoint=100.0)

    assert score(tuned) <= score(seed) + 1e-9    # tuning did not worsen the loop
    assert tuned.kp >= 0.0 and tuned.ki >= 0.0 and tuned.kd >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_numeric_tune.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.cost'`.

- [ ] **Step 3: Implement cost.py and numeric_tune.py**

```python
# src/ragnarok/diagnostics/cost.py
"""Control-tuning cost: ITAE + overshoot + effort penalties (spec §11)."""
from __future__ import annotations

import numpy as np

from ragnarok.diagnostics.metrics import overshoot_pct


def itae_cost(t, measured, command, *, setpoint: float, y0: float = 0.0,
              w_overshoot: float = 1.0, w_effort: float = 0.0) -> float:
    t = np.asarray(t, dtype=float)
    measured = np.asarray(measured, dtype=float)
    command = np.asarray(command, dtype=float)
    dt = float(t[1] - t[0]) if t.size > 1 else 0.0
    itae = float(np.sum(t * np.abs(setpoint - measured) * dt))
    os = overshoot_pct(measured, y0=y0, y_final=setpoint)
    effort = float(np.sum(np.abs(command)))
    return itae + w_overshoot * os + w_effort * effort
```

```python
# src/ragnarok/diagnostics/numeric_tune.py
"""Numeric PID tuning via Nelder-Mead over the synthetic plant (spec §11).

Deterministic: each evaluation builds a fresh FeedbackAimer (PID) + a fresh
plant and scores the closed loop with itae_cost. Returns SEEDS, not final gains.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.diagnostics.plant import simulate_closed_loop
from ragnarok.diagnostics.cost import itae_cost


@dataclass(frozen=True)
class PidSeeds:
    kp: float
    ki: float
    kd: float


def numeric_tune(plant_factory, *, seed: PidSeeds, setpoint: float, n_steps: int,
                 dt_s: float, max_step_px: float = 1e9, w_overshoot: float = 1.0,
                 w_effort: float = 0.0) -> PidSeeds:
    def cost(theta) -> float:
        kp, ki, kd = (max(0.0, float(v)) for v in theta)   # gains are non-negative
        aimer = FeedbackAimer(kp=kp, ki=ki, kd=kd, max_step_px=max_step_px, ema_alpha=1.0)
        t, m, u = simulate_closed_loop(
            lambda e, dt: aimer.step((0.0, 0.0), (e, 0.0), dt)[0],
            plant_factory(), setpoint=setpoint, n_steps=n_steps, dt_s=dt_s,
        )
        return itae_cost(t, m, u, setpoint=setpoint, w_overshoot=w_overshoot, w_effort=w_effort)

    x0 = np.array([seed.kp, seed.ki, seed.kd], dtype=float)
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
    kp, ki, kd = (max(0.0, float(v)) for v in res.x)
    return PidSeeds(kp=kp, ki=ki, kd=kd)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_numeric_tune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/diagnostics/cost.py src/ragnarok/diagnostics/numeric_tune.py tests/diagnostics/test_numeric_tune.py
git commit -m "feat(diagnostics): ITAE cost + Nelder-Mead numeric PID tuner"
```

---

## Task 13: apply_seeds (explicit, never auto-written)

**Files:**
- Create: `src/ragnarok/diagnostics/apply.py`
- Create: `tests/diagnostics/test_apply.py`

**Interfaces:**
- Consumes: `PidSeeds` (T12), `AppConfig`/`AimConfig` (frozen pydantic).
- Produces: `apply_seeds(cfg: AppConfig, seeds: PidSeeds, *, controller_mode="pid") -> AppConfig` — returns a NEW frozen `AppConfig` with `aim.kp/ki/kd` set from `seeds` and `aim.controller_mode` updated, leaving the original untouched. The caller (GUI/CLI, later) is responsible for `ConfigHandle.swap`; nothing here writes to disk or auto-applies (spec §11: seeds, not final).

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_apply.py
"""Tests for apply_seeds (explicit, immutable config update)."""
from __future__ import annotations
from ragnarok.config.schema import AppConfig
from ragnarok.diagnostics.numeric_tune import PidSeeds
from ragnarok.diagnostics.apply import apply_seeds


def test_apply_seeds_returns_new_config_with_gains():
    cfg = AppConfig()
    seeds = PidSeeds(kp=0.5, ki=0.2, kd=0.05)
    out = apply_seeds(cfg, seeds, controller_mode="pid")
    assert out.aim.kp == 0.5 and out.aim.ki == 0.2 and out.aim.kd == 0.05
    assert out.aim.controller_mode == "pid"


def test_original_config_is_unchanged():
    cfg = AppConfig()
    apply_seeds(cfg, PidSeeds(kp=9.9, ki=9.9, kd=9.9))
    assert cfg.aim.kp == 0.35 and cfg.aim.ki == 0.0   # original defaults intact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/diagnostics/test_apply.py -v`
Expected: FAIL — `No module named 'ragnarok.diagnostics.apply'`.

- [ ] **Step 3: Implement apply.py**

```python
# src/ragnarok/diagnostics/apply.py
"""Apply auto-tune SEEDS into a new frozen AppConfig (spec §11: seeds, not final).

Explicit only — never auto-writes config or touches disk. The caller wires the
returned AppConfig into ConfigHandle.swap when (and if) the user accepts it.
"""
from __future__ import annotations

from ragnarok.config.schema import AppConfig
from ragnarok.diagnostics.numeric_tune import PidSeeds


def apply_seeds(cfg: AppConfig, seeds: PidSeeds, *, controller_mode: str = "pid") -> AppConfig:
    new_aim = cfg.aim.model_copy(update={
        "kp": seeds.kp, "ki": seeds.ki, "kd": seeds.kd,
        "controller_mode": controller_mode,
    })
    return cfg.model_copy(update={"aim": new_aim})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/diagnostics/test_apply.py -v`
Expected: PASS

- [ ] **Step 5: Run the FULL suite**

Run: `python -m pytest -q`
Expected: PASS (all prior + all Phase 5A tests).

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/diagnostics/apply.py tests/diagnostics/test_apply.py
git commit -m "feat(diagnostics): explicit apply_seeds into a new frozen AppConfig"
```

---

## Phase 5A completion checklist

- [ ] Measurable: step-response metrics (T1) + resampler (T2) + result/recorder (T3) + plant/sim (T4) + runner (T5) + DiagnosticsConfig (T6).
- [ ] §15 CI regression locks the control loop (T7), green before AND after the PID upgrade.
- [ ] Tunable: FeedbackAimer 2-DOF PID + anti-windup, P-compatible defaults (T8); PID config + wiring with `controller_mode` (T9).
- [ ] Auto-tune: relay + ZN seeding (T10), relay experiment vs plant (T11), ITAE + Nelder-Mead numeric tuner (T12).
- [ ] Explicit, never-auto `apply_seeds` (T13).
- [ ] Full suite green; all live modes (desktop cursor, in-game sensor) behind injected seams (box-only smokes); Scope-Boundary deferrals (5B GMC, 5C dynamic-ROI, GUI, HIL, CMA-ES) documented.

After merge: update memory (Phase 5A done; **Next: Plan 5B — wire FeedForwardGMC into the worker + τ_render/deg_per_count calibration**, reusing `diagnostics/resample.py`). The headless `StepResponseRunner` + `RelayTuneResult`/`PidSeeds` + `apply_seeds` are the contract the Phase-8 Cyberpunk Diagnostics tab will consume.
