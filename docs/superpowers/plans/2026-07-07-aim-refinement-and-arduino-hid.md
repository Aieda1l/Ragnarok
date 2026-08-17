# Aim Refinement + Arduino HID/WiFi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix open-loop aimer overshoot and terminal jitter, make the trigger bot work independently of auto-aim, convert auto-aim and trigger to independent toggles on non-obtrusive hotkeys, make latency calibration reliable, fix the wiring bugs that block Arduino use, add a driverless raw-HID transport, and rewrite the firmware for the UNO R4 WiFi + USB Host Shield passthrough topology.

**Architecture:** In-place behaviour changes at existing seams. Aimers gain a `commit` fraction (open-loop only) + a shared settle deadzone. `AimController` runs the trigger every tick independent of the aim-active gate. `WorkerLoop` latches the latency measurement. `main()` owns one mouse driver. A new `HidTransport` sits behind the existing `ArduinoDriver`. Firmware is box-only.

**Tech Stack:** Python 3.13, pydantic v2 (frozen models), PySide6, pytest, numpy, filterpy; Arduino (Renesas RA4M1 core, `felis/USB_Host_Shield_2.0`), ESP32-S3 (WiFiS3); PC-side `hidapi` (lazy, box-only).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-07-aim-refinement-and-arduino-hid-design.md` (Approach A).
- **CI-safe:** no test may import `torch`, `tensorrt`, `bettercam`, `serial`, `socket` real I/O, `hid`/`hidapi`, PySide6 widgets requiring a display, or touch a real mouse/COM/MCU. All hardware is injected as a fake.
- **TDD:** write the failing test first; run it red; implement minimally; run it green; commit.
- **Frozen config:** `AppConfig` and all sub-models are `ConfigDict(frozen=True)`. Edit via `model_copy(update=...)` **only after** re-validating through the sub-model class (see `tuning_model.set_field`).
- **Back-compat:** new config fields must have defaults; `commit=1.0` must reproduce prior open-loop aimer output; `FeedbackAimer` must stay byte-identical at its current defaults (it never consumes `commit`); `TriggerBot` with `max_occlusion_frames=0` must reproduce the old `not occluded` behaviour.
- **Single class:** `Team.ENEMY`-only targeting is a safety contract — never widen it in code; "shoot anything" is achieved by disabling friend/foe (which stamps all `ENEMY`).
- **Test runner:** `uv run python -m pytest -p no:cacheprovider <path> -q` (foreground; do not use bare `uv run pytest`).
- **Commit trailers:** end each commit body with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01RUNWZw1D6mjjNmbanTF74o`.
- **Branch:** `phase-9p-aim-refinement-arduino`.

---

## File Structure

- `src/ragnarok/config/schema.py` — new aim fields (`commit`, `settle_px`), changed defaults (`adaptive_lead`, `lead_ms`, `toggle`, `aim_key`); trigger changes (`trigger_key`, `activation_delay_ms`, `max_occlusion_frames`); `ArduinoConfig.transport` gains `"hid"` + `vid`/`hid_pid`.
- `src/ragnarok/aim/aimers.py` — `commit` + `settle_px` on the three open-loop aimers; `settle_px` on feedback; `PredictiveAimer` remaining-distance clamp.
- `src/ragnarok/aim/controller.py` — trigger runs every tick independent of aim; crosshair-containment target; recoil in the trigger path.
- `src/ragnarok/trigger/bot.py` — `max_occlusion_frames` tolerance.
- `src/ragnarok/trigger/controller.py` + `tests/aim/test_controller.py` — **removed** (unified into `AimController`).
- `src/ragnarok/telemetry/snapshot.py` — `aim_on` / `trigger_on` fields.
- `src/ragnarok/worker/loop.py` — latency latch, detector single-read/TOCTOU, publish toggle state.
- `src/ragnarok/capture/bettercam_capturer.py` — arrival-time stamp (box-only real path) + injectable clock.
- `src/ragnarok/gui/worker_thread.py` — stop the capturer before join.
- `src/ragnarok/gui/live_config.py` — close-old-driver-before-new + rollback.
- `src/ragnarok/aim/arduino.py` — `HidTransport`, `build_arduino_transport` routing.
- `src/ragnarok/gui/tuning_model.py` — trigger/keybind/arduino field updates.
- `src/ragnarok/app.py` — single-owner mouse driver, unified fire component, overlay/dashboard state.
- `src/ragnarok/gui/dashboard_panel.py`, `gui/overlay_window.py` — AIM/TRIGGER ON/OFF indicators (box-only render).
- `firmware/ragnarok_mouse_r4/ragnarok_mouse_r4.ino` — RA4M1 passthrough firmware (box-only).
- `firmware/ragnarok_esp32_udp/ragnarok_esp32_udp.ino` — ESP32-S3 UDP→UART firmware (box-only).

---

### Task 1: Aimer `commit` fraction + shared settle deadzone

**Files:**
- Modify: `src/ragnarok/config/schema.py` (AimConfig: add `commit`, `settle_px`)
- Modify: `src/ragnarok/aim/aimers.py`
- Modify: `src/ragnarok/wiring.py` (`build_aimer`)
- Test: `tests/aim/test_aimers.py`, `tests/aim/test_hybrid_predictive.py`

**Interfaces:**
- Produces:
  - `FlickAimer(*, flick_speed_px_s, settle_px=0.0)`
  - `HybridAimer(*, kp, max_step_px, flick_dist_px, flick_speed_px_s, ema_alpha=1.0, commit=1.0, settle_px=0.0)`
  - `PredictiveAimer(*, max_step_px, kff=1.0, commit=1.0, settle_px=0.0)`
  - `FeedbackAimer(*, kp, max_step_px, ema_alpha=1.0, kff=0.0, ki=0.0, kd=0.0, integral_clamp=None, cond_integ_thresh_px=None, creep_px=0.0, settle_px=0.0)`
  - `AimConfig.commit: float` (default 0.85), `AimConfig.settle_px: float` (default 2.0)

- [ ] **Step 1: Add the config fields.** In `schema.py`, inside `AimConfig`, after `creep_px`:

```python
    commit: float = Field(default=0.85, gt=0.0, le=1.0)   # open-loop aimers issue commit*step
    settle_px: float = Field(default=2.0, ge=0.0)         # deadzone: <= this error -> no move
```

- [ ] **Step 2: Write failing tests** in `tests/aim/test_aimers.py` (append):

```python
from ragnarok.aim.aimers import HybridAimer, PredictiveAimer


def test_settle_deadzone_zeroes_small_error_flick():
    a = FlickAimer(flick_speed_px_s=1000.0, settle_px=3.0)
    assert a.step((0.0, 0.0), (2.0, 0.0), dt=1.0) == (0.0, 0.0)   # 2px <= 3px -> hold


def test_flick_moves_outside_settle():
    a = FlickAimer(flick_speed_px_s=1000.0, settle_px=3.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), dt=1.0)
    assert abs(dx - 10.0) < 1e-6 and abs(dy) < 1e-6


def test_predictive_clamps_to_remaining_not_max_step():
    # error 5px, max_step 60, commit 1.0 -> must NOT exceed remaining distance
    a = PredictiveAimer(max_step_px=60.0, kff=0.0, commit=1.0)
    dx, dy = a.step((0.0, 0.0), (5.0, 0.0), dt=0.016)
    assert abs(dx - 5.0) < 1e-6, f"overshoot dx={dx}"


def test_predictive_commit_scales_step():
    a = PredictiveAimer(max_step_px=60.0, kff=0.0, commit=0.5)
    dx, dy = a.step((0.0, 0.0), (40.0, 0.0), dt=0.016)   # remaining 40, commit .5 -> 20
    assert abs(dx - 20.0) < 1e-6, f"dx={dx}"


def test_predictive_commit_one_reproduces_full_step_within_max():
    a = PredictiveAimer(max_step_px=60.0, kff=0.0, commit=1.0)
    dx, dy = a.step((0.0, 0.0), (30.0, 0.0), dt=0.016)
    assert abs(dx - 30.0) < 1e-6


def test_hybrid_close_leg_commit_fraction():
    # inside flick_dist (20): snap the remaining error * commit
    a = HybridAimer(kp=0.35, max_step_px=60.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, commit=0.5)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), dt=0.016)
    assert abs(dx - 5.0) < 1e-6, f"dx={dx}"


def test_feedback_ignores_commit_and_stays_identical():
    # FeedbackAimer has no commit param; settle_px default 0 -> byte-identical
    a = FeedbackAimer(kp=0.35, max_step_px=60.0)
    dx, dy = a.step((0.0, 0.0), (100.0, 0.0), dt=0.016)
    assert abs(dx - 35.0) < 1e-6   # kp*100, unchanged
```

- [ ] **Step 3: Run red.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_aimers.py -q` → FAIL (unexpected kwargs / wrong clamp).

- [ ] **Step 4: Implement.** In `aimers.py`:

Add a module-level helper after the imports:

```python
def _settled(ex: float, ey: float, settle_px: float) -> bool:
    """True when the crosshair is within the settle deadzone of the target."""
    return settle_px > 0.0 and math.hypot(ex, ey) <= settle_px
```

`FlickAimer.__init__`: add `settle_px: float = 0.0` and store `self._settle = settle_px`. In `step`, after computing `ex, ey`:

```python
        if _settled(ex, ey, self._settle):
            return (0.0, 0.0)
```

`HybridAimer.__init__`: add `commit: float = 1.0, settle_px: float = 0.0`; store `self._commit`, `self._settle`. In `step`, after `ex, ey, d`:

```python
        if _settled(ex, ey, self._settle):
            return (0.0, 0.0)
```
Change the close-leg return from `return (ex, ey)` to `return (ex * self._commit, ey * self._commit)`. Multiply the far-leg `dx, dy` by `self._commit` right before returning.

`PredictiveAimer.__init__`: add `commit: float = 1.0, settle_px: float = 0.0`; store them. Rewrite `step` body:

```python
        ex = target_point[0] - crosshair[0]
        ey = target_point[1] - crosshair[1]
        if _settled(ex, ey, self._settle):
            return (0.0, 0.0)
        dx = ex + self._kff * target_vel[0] * dt
        dy = ey + self._kff * target_vel[1] * dt
        mag = math.hypot(dx, dy)
        remaining = math.hypot(ex, ey)
        limit = min(self._max, remaining)          # never overshoot remaining OR max step
        if mag > limit and mag > 0.0:
            s = limit / mag
            dx *= s
            dy *= s
        return (dx * self._commit, dy * self._commit)
```

`FeedbackAimer.__init__`: add `settle_px: float = 0.0` (last param); store `self._settle`. At the very start of `step` (before the sign-flip block), add:

```python
        if _settled(target_point[0] - crosshair[0],
                    target_point[1] - crosshair[1], self._settle):
            return (0.0, 0.0)
```

- [ ] **Step 5: Wire `build_aimer`.** In `wiring.py`, update each branch:

```python
    if a.aimer == "flick":
        return FlickAimer(flick_speed_px_s=a.flick_speed_px_s, settle_px=a.settle_px)
    if a.aimer == "hybrid":
        return HybridAimer(
            kp=a.kp, max_step_px=a.max_step_px,
            flick_dist_px=a.hybrid_flick_dist_px,
            flick_speed_px_s=a.flick_speed_px_s, ema_alpha=a.ema_alpha,
            commit=a.commit, settle_px=a.settle_px,
        )
    if a.aimer == "predictive":
        return PredictiveAimer(max_step_px=a.max_step_px, kff=a.kff,
                               commit=a.commit, settle_px=a.settle_px)
    ki = a.ki if a.controller_mode in ("pi", "pid") else 0.0
    kd = a.kd if a.controller_mode == "pid" else 0.0
    return FeedbackAimer(
        kp=a.kp, max_step_px=a.max_step_px, ema_alpha=a.ema_alpha, kff=a.kff,
        ki=ki, kd=kd, integral_clamp=a.integral_clamp,
        cond_integ_thresh_px=a.cond_integ_thresh_px, creep_px=a.creep_px,
        settle_px=a.settle_px,
    )
```

- [ ] **Step 6: Run green + full aim suite.** `uv run python -m pytest -p no:cacheprovider tests/aim -q` → PASS. Confirm `tests/aim/test_smith_predictor.py`, `test_feedback_pid.py`, `test_feedback_damping.py`, `test_hybrid_predictive.py` still pass (settle default 0 in those constructors keeps them identical).

- [ ] **Step 7: Add the Aim-tab fields** to `gui/tuning_model.py` `AIM_FIELDS` (after `creep_px`):

```python
    FieldSpec("aim.commit", "Commit fraction", "float", 0.05, 1.0, 0.05),
    FieldSpec("aim.settle_px", "Settle deadzone (px)", "float", 0.0, 20.0, 0.5),
```

- [ ] **Step 8: Commit.**

```bash
git add src/ragnarok/config/schema.py src/ragnarok/aim/aimers.py src/ragnarok/wiring.py src/ragnarok/gui/tuning_model.py tests/aim/test_aimers.py
git commit -m "feat(aim): commit fraction + settle deadzone (kills open-loop overshoot/jitter)"
```

---

### Task 2: Snappy/steady schema defaults

**Files:**
- Modify: `src/ragnarok/config/schema.py` (AimConfig defaults)
- Test: `tests/config/test_schema_defaults.py` (create)

**Interfaces:**
- Produces: `AimConfig().adaptive_lead == False`, `AimConfig().lead_ms == 0.0`.

- [ ] **Step 1: Write the failing test.** Create `tests/config/test_schema_defaults.py`:

```python
from ragnarok.config.schema import AimConfig, TriggerConfig


def test_snappy_aim_defaults():
    a = AimConfig()
    assert a.adaptive_lead is False      # no lead-induced jitter by default
    assert a.lead_ms == 0.0
    assert a.commit == 0.85
    assert a.settle_px == 2.0


def test_aim_toggle_defaults():
    a = AimConfig()
    assert a.toggle is True               # toggle, not hold
    assert a.aim_key == "VK_XBUTTON2"     # non-obtrusive default


def test_trigger_defaults():
    t = TriggerConfig()
    assert t.trigger_key == "VK_XBUTTON1"
    assert t.activation_delay_ms == 35.0
    assert t.max_occlusion_frames == 2
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/config/test_schema_defaults.py -q` → FAIL.

- [ ] **Step 3: Change the defaults** in `schema.py`. `AimConfig`: `adaptive_lead: bool = False`; `lead_ms: float = Field(default=0.0, ge=0.0, le=500.0)`; `toggle: bool = True`; `aim_key: str = "VK_XBUTTON2"`. (`TriggerConfig` fields land in Task 5 — this test's trigger assertions stay red until then; mark it `@pytest.mark.xfail(reason="trigger defaults land in Task 5")` on `test_trigger_defaults`, or split the trigger asserts into Task 5. Prefer: move `test_trigger_defaults` to Task 5 and keep only the two aim tests here.)

Keep only the two aim tests in this task; add `test_trigger_defaults` in Task 5.

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/config/test_schema_defaults.py -q` → PASS (2 tests).

- [ ] **Step 5: Regression sweep.** `uv run python -m pytest -p no:cacheprovider -q` → PASS. If any test asserted the old `adaptive_lead=True`/`lead_ms=40`/`toggle=False`/`aim_key="VK_RBUTTON"` default, update it to the new value (search `tests/` for those literals first).

- [ ] **Step 6: Commit.**

```bash
git add src/ragnarok/config/schema.py tests/config/test_schema_defaults.py
git commit -m "feat(aim): snappy/steady defaults (adaptive_lead off, lead_ms 0, aim toggle)"
```

---

### Task 3: TriggerBot occlusion tolerance

**Files:**
- Modify: `src/ragnarok/trigger/bot.py`
- Test: `tests/trigger/test_bot.py`

**Interfaces:**
- Produces: `TriggerBot(*, mouse, activation_delay_s, button=MouseButton.LEFT, max_occlusion_frames=0, clock=now_ns)`. With `max_occlusion_frames=0`, behaviour is identical to today.

- [ ] **Step 1: Write failing tests** (append to `tests/trigger/test_bot.py`):

```python
def test_brief_occlusion_does_not_reset_eligibility():
    from ragnarok.trigger.bot import TriggerBot
    from ragnarok.core.types import Track, Team
    from ragnarok.aim.mouse import MouseButton

    class _M:
        def __init__(self): self.buttons = []
        def set_button(self, b, d): self.buttons.append((b, d))

    t = [0]
    def clk(): return t[0]
    tr = Track(track_id=1, xyxy=(0, 0, 10, 10), confidence=0.9, class_id=0, team=Team.ENEMY)
    bot = TriggerBot(mouse=_M(), activation_delay_s=0.05, max_occlusion_frames=2, clock=clk)
    kw = dict(track=tr, crosshair=(5, 5), enemy_confirmed=True, line_clear=True, active=True)
    bot.update(occluded=False, **kw)          # eligible at t=0
    t[0] = int(0.03e9)
    bot.update(occluded=True, **kw)           # 1 occluded frame — tolerated
    t[0] = int(0.06e9)
    fired = bot.update(occluded=True, **kw)   # 2 occluded frames — still tolerated, delay elapsed
    assert fired is True


def test_occlusion_beyond_tolerance_resets():
    from ragnarok.trigger.bot import TriggerBot
    from ragnarok.core.types import Track, Team

    class _M:
        def set_button(self, b, d): pass

    tr = Track(track_id=1, xyxy=(0, 0, 10, 10), confidence=0.9, class_id=0, team=Team.ENEMY)
    bot = TriggerBot(mouse=_M(), activation_delay_s=0.0, max_occlusion_frames=1)
    kw = dict(track=tr, crosshair=(5, 5), enemy_confirmed=True, line_clear=True, active=True)
    bot.update(occluded=True, **kw)
    bot.update(occluded=True, **kw)           # 2 consecutive > tolerance(1) -> not ready
    assert bot.is_firing is False
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/trigger/test_bot.py -q` → FAIL (unexpected `max_occlusion_frames`).

- [ ] **Step 3: Implement.** In `bot.py` `__init__` add `max_occlusion_frames: int = 0` and `self._max_occ = max_occlusion_frames`, `self._occ_streak = 0`. Rewrite the readiness computation in `update`:

```python
    def update(self, *, track, crosshair, occluded, enemy_confirmed, line_clear, active) -> bool:
        self._occ_streak = self._occ_streak + 1 if occluded else 0
        ready = (
            active
            and enemy_confirmed
            and line_clear
            and self._occ_streak <= self._max_occ
            and track is not None
            and self._inside(track, crosshair)
        )
        if not ready:
            self._eligible_since = None
            self._release_if_pressed()
            return False
        now = self._clock()
        if self._eligible_since is None:
            self._eligible_since = now
        elapsed = (now - self._eligible_since) / 1e9
        if elapsed >= self._delay and not self._pressed:
            self._mouse.set_button(self._button, True)
            self._pressed = True
            return True
        return False
```

Reset `self._occ_streak = 0` in `release()`.

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/trigger/test_bot.py -q` → PASS. (Existing `not occluded` tests pass because default `max_occlusion_frames=0`.)

- [ ] **Step 5: Commit.**

```bash
git add src/ragnarok/trigger/bot.py tests/trigger/test_bot.py
git commit -m "feat(trigger): tolerate brief occlusion so flickery detections still fire"
```

---

### Task 4: Unify trigger into AimController — independent of aim key

**Files:**
- Modify: `src/ragnarok/aim/controller.py`
- Remove: `src/ragnarok/trigger/controller.py`, `tests/aim/test_controller.py` (the standalone path is now `AimController` with aim disabled)
- Test: `tests/aim/test_controller_phase4.py` (append) and a new `tests/aim/test_trigger_independent.py`

**Interfaces:**
- Consumes: `TriggerBot.update(...)` (Task 3), `RecoilCompensator.on_fire()/release()`, `CommandedMotionBuffer.push`.
- Produces: `AimController` fires the trigger every tick when a trigger is present and `trigger_active()` is true, regardless of `aim_active()`; exposes `self.target_id` (aim lock) unchanged and a new `self.fire_target_id` (crosshair-contained enemy).

- [ ] **Step 1: Write failing test.** Create `tests/aim/test_trigger_independent.py`:

```python
from ragnarok.aim.controller import AimController
from ragnarok.aim.imm import IMMManager
from ragnarok.config.schema import AimConfig
from ragnarok.core.types import Track, Tracks, Team


class _Mouse:
    def __init__(self): self.moves = []; self.buttons = []
    def move_relative(self, dx, dy): self.moves.append((dx, dy))
    def set_button(self, b, d): self.buttons.append((b, d))


class _Sel:
    def select(self, tracks, cx, cy): return None
    def reset(self): pass


class _Trig:
    def __init__(self): self.calls = []; self.is_firing = False
    def update(self, **kw): self.calls.append(kw); return True
    def release(self): pass


def _enemy_center(roi=100):
    c = roi / 2.0
    return Track(track_id=3, xyxy=(c-10, c-10, c+10, c+10), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _cfg(enabled):
    return AimConfig(enabled=enabled, hfov_deg=90.0, screen_width_px=900, sensitivity=1.0)


def test_trigger_fires_with_aim_disabled():
    trig = _Trig()
    c = AimController(_cfg(enabled=False), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: False,
                      roi_size=100, trigger=trig, trigger_active=lambda: True)
    c.update(Tracks(items=(_enemy_center(),)), t_capture_ns=0)
    assert len(trig.calls) == 1 and trig.calls[0]["active"] is True
    assert c.fire_target_id == 3


def test_trigger_fires_with_aim_enabled_but_key_up():
    trig = _Trig()
    c = AimController(_cfg(enabled=True), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: False,   # aim key UP
                      roi_size=100, trigger=trig, trigger_active=lambda: True)
    c.update(Tracks(items=(_enemy_center(),)), t_capture_ns=0)
    assert len(trig.calls) == 1               # trigger still fired despite aim key up
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_trigger_independent.py -q` → FAIL (trigger not called when aim inactive).

- [ ] **Step 3: Restructure `AimController`.** In `controller.py`:

Add in `__init__` (near `self.target_id`): `self.fire_target_id: int | None = None`.

Replace the top of `update` so the trigger runs before the aim-active gate:

```python
    def update(self, tracks: Tracks, t_capture_ns: int) -> None:
        self._imm.prune({t.track_id for t in tracks})

        # TRIGGER: evaluated every tick, independent of the aim key/toggle.
        self._run_trigger(tracks)

        # AIM ASSIST: only while aim is enabled AND its toggle is active.
        if not (self._cfg.enabled and self._active()):
            self._disengage_aim()
            return
        # ... existing selection / lead / aimer / move body unchanged ...
```

Move the existing trigger+recoil block **out** of the aim body into a new method, and delete it from the aim body (the aim body keeps only selection, IMM, lead, aimer.step, shaper, and the aim move + cmd_buf push):

```python
    def _run_trigger(self, tracks: Tracks) -> None:
        if self._trigger is None:
            self.fire_target_id = None
            return
        target = self._enemy_under_crosshair(tracks)
        self.fire_target_id = target.track_id if target is not None else None
        if target is None:
            self._trigger.release()
            if self._recoil is not None:
                self._recoil.release()
            self._was_firing = False
            return
        fired = self._trigger.update(
            track=target, crosshair=(self._cx, self._cy),
            occluded=target.time_since_update > 0, enemy_confirmed=True,
            line_clear=self._line_clear(), active=self._trigger_active(),
        )
        if self._recoil is not None:
            rx, ry = self._recoil_delta(fired)
            if rx or ry:
                k = self._deg_per_px / self._cfg.sensitivity
                cdx, cdy = rx * k, ry * k
                self._mouse.move_relative(cdx, cdy)
                if self._cmd_buf is not None:
                    self._cmd_buf.push(self._clock(), cdx, cdy)

    def _recoil_delta(self, fired: bool) -> tuple[float, float]:
        now = self._clock()
        rps = getattr(self._recoil, "fire_rate_rps", 0.0)
        firing = self._trigger.is_firing
        rx = ry = 0.0
        if self._was_firing and not firing:
            self._recoil.release()
        if fired:
            rx, ry = self._recoil.on_fire()
            self._last_shot_ns = now
        elif rps > 0.0 and firing and now - self._last_shot_ns >= 1e9 / rps:
            rx, ry = self._recoil.on_fire()
            self._last_shot_ns = now
        self._was_firing = firing
        return rx, ry

    @staticmethod
    def _enemy_under_crosshair(tracks: Tracks):
        return None  # replaced below

    def _disengage_aim(self) -> None:
        self._aimer.reset() if self._aimer is not None else None
        self._shaper.reset()
        if self._vel is not None:
            self._vel.reset()
        self._sel.reset()
        self._last_ns = None
        self._cur_target = None
        self.target_id = None
```

Implement `_enemy_under_crosshair` as an instance method (replace the placeholder staticmethod):

```python
    def _enemy_under_crosshair(self, tracks: Tracks):
        from ragnarok.core.types import Team
        for t in tracks:
            if t.team is Team.ENEMY:
                x1, y1, x2, y2 = t.xyxy
                if x1 <= self._cx <= x2 and y1 <= self._cy <= y2:
                    return t
        return None
```

In the aim body, delete the old `if self._trigger is not None:` fire/recoil block (now handled by `_run_trigger`); keep the aim move + `cmd_buf.push`. Guard the aimer call for `aimer is None` is unnecessary (aim body only runs when enabled and a real aimer is wired), but `_disengage_aim` must tolerate `self._aimer is None` (shown above). Keep the old `_reset_stateful`/`_disengage` only if still referenced; otherwise remove them and route `_disengage()`→`_disengage_aim()` plus a trigger/recoil release.

> Note: the aim body previously reset the trigger via `_reset_stateful`. Since the trigger is now independent, do NOT release it on aim disengage. On full controller teardown (target lost, key up) the trigger is released inside `_run_trigger` when no enemy is under the crosshair.

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_trigger_independent.py tests/aim/test_controller_phase4.py tests/aim/test_smith_predictor.py tests/aim/test_recoil_fullauto.py -q` → PASS. Fix any references to removed methods.

- [ ] **Step 5: Remove the standalone TriggerController.**

```bash
git rm src/ragnarok/trigger/controller.py tests/aim/test_controller.py
```

Grep for imports: `uv run python -m pytest -p no:cacheprovider -q` and fix any `from ragnarok.trigger.controller import TriggerController` (app.py handled in Task 10).

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "feat(trigger): unify into AimController, fire independent of aim key"
```

---

### Task 5: Toggle activation for aim + trigger, config + keybinds

**Files:**
- Modify: `src/ragnarok/config/schema.py` (TriggerConfig)
- Modify: `src/ragnarok/gui/tuning_model.py` (TRIGGER_FIELDS, KEYBIND_FIELDS)
- Test: `tests/config/test_schema_defaults.py` (add `test_trigger_defaults` from Task 2)

**Interfaces:**
- Produces: `TriggerConfig.trigger_key` default `"VK_XBUTTON1"`, `activation_delay_ms` default `35.0`, `max_occlusion_frames: int` (default 2).

- [ ] **Step 1: Add the trigger defaults test** (moved from Task 2) to `tests/config/test_schema_defaults.py`:

```python
def test_trigger_defaults():
    from ragnarok.config.schema import TriggerConfig
    t = TriggerConfig()
    assert t.trigger_key == "VK_XBUTTON1"
    assert t.activation_delay_ms == 35.0
    assert t.max_occlusion_frames == 2
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/config/test_schema_defaults.py::test_trigger_defaults -q` → FAIL.

- [ ] **Step 3: Implement.** In `schema.py` `TriggerConfig`: `trigger_key: str = "VK_XBUTTON1"`; `activation_delay_ms: float = Field(default=35.0, ge=0.0, le=2000.0)`; add `max_occlusion_frames: int = Field(default=2, ge=0, le=30)`.

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/config/test_schema_defaults.py -q` → PASS.

- [ ] **Step 5: Surface in the GUI.** In `tuning_model.py`, add to `TRIGGER_FIELDS`:

```python
    FieldSpec("trigger.trigger_key", "Trigger key (VK_ or char)", "text"),
    FieldSpec("trigger.max_occlusion_frames", "Occlusion tolerance (frames)", "int", 0, 30, 1),
```

(Keep `trigger.trigger_key` here on the Fire tab AND in `KEYBIND_FIELDS`; both edit the same field. If duplication is undesirable, leave it only in `KEYBIND_FIELDS` — pick one and note it.) Update the `KEYBIND_FIELDS` label for `aim.aim_key` to `"Aim toggle key (VK_ or char)"` and `aim.toggle` to `"Aim = toggle"`.

- [ ] **Step 6: Build trigger_active in toggle mode.** (Deferred to Task 10's `_build_trigger_bot` change, since it lives in `app.py`.) Note here so it's not missed: `trigger_active = make_aim_active(AsyncKeyStateProvider(cfg.trigger.trigger_key), toggle=True)`.

- [ ] **Step 7: Commit.**

```bash
git add src/ragnarok/config/schema.py src/ragnarok/gui/tuning_model.py tests/config/test_schema_defaults.py
git commit -m "feat(trigger): toggle activation + non-obtrusive default keys"
```

---

### Task 6: Toggle-state telemetry (AIM/TRIGGER ON/OFF)

**Files:**
- Modify: `src/ragnarok/telemetry/snapshot.py`
- Modify: `src/ragnarok/aim/controller.py` (expose live state)
- Modify: `src/ragnarok/worker/loop.py` (publish it)
- Test: `tests/telemetry/test_snapshot.py` (append or create), `tests/aim/test_trigger_independent.py`

**Interfaces:**
- Produces: `TelemetrySnapshot.aim_on: bool | None`, `.trigger_on: bool | None`; `AimController.aim_on: bool`, `AimController.trigger_on: bool`.

- [ ] **Step 1: Write failing test** (append to `tests/aim/test_trigger_independent.py`):

```python
def test_controller_exposes_toggle_state():
    trig = _Trig()
    c = AimController(_cfg(enabled=True), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: True,
                      roi_size=100, trigger=trig, trigger_active=lambda: False)
    c.update(Tracks(items=()), t_capture_ns=0)
    assert c.aim_on is True and c.trigger_on is False
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_trigger_independent.py::test_controller_exposes_toggle_state -q` → FAIL.

- [ ] **Step 3: Implement.** In `AimController.__init__`: `self.aim_on = False`, `self.trigger_on = False`. In `update`, set them from the live closures near the top (after `_run_trigger`):

```python
        self.aim_on = bool(self._cfg.enabled and self._active())
        self.trigger_on = bool(self._trigger is not None and self._trigger_active())
```

- [ ] **Step 4: Add snapshot fields.** In `snapshot.py` append to `TelemetrySnapshot`:

```python
    aim_on: bool | None = None                # Phase 9P: auto-aim toggle state
    trigger_on: bool | None = None            # Phase 9P: trigger toggle state
```

- [ ] **Step 5: Publish from the loop.** In `loop.py` `tick`, in the `self._pub.publish(TelemetrySnapshot(...))` call add:

```python
            aim_on=getattr(aim, "aim_on", None),
            trigger_on=getattr(aim, "trigger_on", None),
```

- [ ] **Step 6: Run green.** `uv run python -m pytest -p no:cacheprovider tests/aim tests/telemetry -q` → PASS.

- [ ] **Step 7: Render the indicators (box-only wiring).** In `gui/dashboard_panel.py`, add two `QLabel`s that read `publisher.latest().aim_on/.trigger_on` on its existing timer and show `AIM: ON`/`OFF` (cyan/red per `theme`) and `TRIGGER: ON`/`OFF`. In `gui/overlay_window.py`, draw the same two words top-left. These are visual/box-only; keep them behind the existing paint timers. No new unit test (rendering is box-only), but keep the read defensive (`snap is None` guard).

- [ ] **Step 8: Commit.**

```bash
git add src/ragnarok/telemetry/snapshot.py src/ragnarok/aim/controller.py src/ragnarok/worker/loop.py src/ragnarok/gui/dashboard_panel.py src/ragnarok/gui/overlay_window.py tests/aim/test_trigger_independent.py
git commit -m "feat(gui): publish + show AIM/TRIGGER on-off toggle state"
```

---

### Task 7: Latency-measure latch

**Files:**
- Modify: `src/ragnarok/worker/loop.py`
- Test: `tests/worker/test_loop_latency_latch.py` (create)

**Interfaces:**
- Produces: `latency_ms` persists in every published snapshot until the next `request_latency_measure`.

- [ ] **Step 1: Write failing test.** Create `tests/worker/test_loop_latency_latch.py`:

```python
import numpy as np
from ragnarok.worker.loop import WorkerLoop
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.core.types import Frame, Detections


class _Cap:
    def start(self): pass
    def stop(self): pass
    def grab(self):
        return Frame(image=np.zeros((8, 8, 3), np.uint8), t_capture_ns=0, region=(0, 0, 8, 8))


class _Det:
    def detect(self, frame): return Detections.empty()
    def set_confidence(self, c): pass


class _Measurer:
    def __init__(self, *a, **k): pass
    def run(self): return 0.042        # 42 ms


def test_latency_latched_across_ticks(monkeypatch):
    import ragnarok.worker.loop as loopmod
    monkeypatch.setattr(loopmod, "WallLatencyMeasurer", _Measurer)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.set_measure_mouse(object())
    loop.request_latency_measure(0.1)
    loop.tick()                                   # runs the measurement
    assert pub.latest().latency_ms == 42.0
    loop.tick()                                   # a normal tick later
    assert pub.latest().latency_ms == 42.0        # STILL latched (was None before)


def test_latency_cleared_on_new_request(monkeypatch):
    import ragnarok.worker.loop as loopmod
    monkeypatch.setattr(loopmod, "WallLatencyMeasurer", _Measurer)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub)
    loop.set_measure_mouse(object())
    loop.request_latency_measure(0.1)
    loop.tick()
    assert pub.latest().latency_ms == 42.0
    loop.request_latency_measure(0.1)             # new request resets the latch
    loop.tick()
    assert pub.latest().latency_ms == 42.0        # fresh measurement re-latched
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/worker/test_loop_latency_latch.py -q` → FAIL (second tick sees `None`).

- [ ] **Step 3: Implement.** In `loop.py`: in `request_latency_measure`, also clear the latch: `self._measure_ms = None`. In `tick`, delete the final `self._measure_ms = None` line (do NOT clear after publishing). The measurement block already sets `self._measure_ms` only when a request is consumed, so it persists across normal ticks and is overwritten only by the next measurement.

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/worker -q` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/ragnarok/worker/loop.py tests/worker/test_loop_latency_latch.py
git commit -m "fix(calibrate): latch measured latency so the GUI reliably applies it"
```

---

### Task 8: Detector single-read (TOCTOU) + clean shutdown

**Files:**
- Modify: `src/ragnarok/worker/loop.py`
- Modify: `src/ragnarok/gui/worker_thread.py`
- Test: `tests/worker/test_loop_detector_swap.py` (create)

**Interfaces:**
- Produces: `tick()` reads `self._det` exactly once; `WorkerThread.stop()` stops the capturer before `wait()`.

- [ ] **Step 1: Write failing test.** Create `tests/worker/test_loop_detector_swap.py`:

```python
import numpy as np
from ragnarok.worker.loop import WorkerLoop
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.core.types import Frame, Detections


class _Cap:
    def start(self): pass
    def stop(self): pass
    def grab(self):
        return Frame(image=np.zeros((8, 8, 3), np.uint8), t_capture_ns=0, region=(0, 0, 8, 8))


class _RoiDet:
    """Detector WITH observe_lock that swaps itself for a plain one mid-detect."""
    def __init__(self, loop): self._loop = loop
    def detect(self, frame):
        self._loop.set_detector(_PlainDet())   # swap to a detector w/o observe_lock
        return Detections.empty()
    def set_confidence(self, c): pass
    def observe_lock(self, center, locked): pass


class _PlainDet:
    def detect(self, frame): return Detections.empty()
    def set_confidence(self, c): pass


def test_tick_single_reads_detector_no_crash():
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), None, StageProfiler(), pub)
    loop.set_detector(_RoiDet(loop))
    loop.tick()      # must NOT raise AttributeError (observe_lock resolved on the swapped-in detector)
    assert pub.latest() is not None
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/worker/test_loop_detector_swap.py -q` → FAIL (AttributeError: `_PlainDet` has no `observe_lock`).

- [ ] **Step 3: Implement.** In `loop.py` `tick`, snapshot the detector once at the top (after the latency block):

```python
        det = self._det
        ...
        dets = det.detect(frame)
        ...
        if hasattr(det, "observe_lock"):
            tid = getattr(aim, "target_id", None) if aim is not None else None
            locked = next((t for t in tracks if t.track_id == tid), None) if tid is not None else None
            det.observe_lock(locked.center if locked is not None else None, locked is not None)
```

Wrap the whole `run` loop body so a swap-induced exception logs instead of killing the thread silently:

```python
    def run(self, stop_event):
        self._cap.start()
        try:
            while not stop_event.is_set():
                try:
                    self.tick()
                except Exception:
                    import traceback, warnings
                    warnings.warn("worker tick failed:\n" + traceback.format_exc())
        finally:
            self._cap.stop()
```

- [ ] **Step 4: Clean shutdown.** In `worker_thread.py` `stop`:

```python
    def stop(self) -> None:
        self._stop.set()
        try:
            self._loop._cap.stop()   # unblock a waiting capturer so run() can exit
        except Exception:
            pass
        self.wait(2000)
```

- [ ] **Step 5: Run green.** `uv run python -m pytest -p no:cacheprovider tests/worker -q` → PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/ragnarok/worker/loop.py src/ragnarok/gui/worker_thread.py tests/worker/test_loop_detector_swap.py
git commit -m "fix(worker): single-read detector (TOCTOU) + stop capturer on shutdown"
```

---

### Task 9: Capture arrival-time timestamp (injectable clock; DXGI box-only)

**Files:**
- Modify: `src/ragnarok/capture/bettercam_capturer.py`
- Test: `tests/capture/test_bettercam_timestamp.py` (create)

**Interfaces:**
- Produces: `BetterCamCapturer(config, screen_size, *, bettercam_module=None, clock=now_ns)`; `grab()` stamps `Frame.t_capture_ns` from `clock()`.

- [ ] **Step 1: Write failing test.** Create `tests/capture/test_bettercam_timestamp.py`:

```python
import numpy as np
from ragnarok.capture.bettercam_capturer import BetterCamCapturer
from ragnarok.config.schema import CaptureConfig


class _FakeCam:
    def start(self, **k): pass
    def stop(self): pass
    def get_latest_frame(self): return np.zeros((4, 4, 3), np.uint8)


class _FakeMod:
    def create(self, **k): return _FakeCam()


def test_grab_uses_injected_clock():
    ticks = iter([1234, 5678])
    cap = BetterCamCapturer(CaptureConfig(), (100, 100),
                            bettercam_module=_FakeMod(), clock=lambda: next(ticks))
    cap.start()
    f = cap.grab()
    assert f.t_capture_ns == 1234
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/capture/test_bettercam_timestamp.py -q` → FAIL (unexpected `clock` kwarg).

- [ ] **Step 3: Implement.** In `bettercam_capturer.py` `__init__` add `clock=now_ns` and `self._clock = clock`. In `grab`, replace `t_capture_ns=now_ns()` with `t_capture_ns=self._clock()`. Add a box-only comment documenting the real fix:

```python
        # BOX-ONLY REFINEMENT: for true arrival time, source t from the DXGI
        # DXGI_OUTDUPL_FRAME_INFO.LastPresentTime (QPC) exposed by the bettercam
        # duplicator instead of the consume-time clock; this removes the 0..1-frame
        # buffer-age jitter that leaks into IMM velocity. The injectable clock here
        # is the CI seam; the real DXGI wiring is verified on the box.
```

- [ ] **Step 4: Run green.** `uv run python -m pytest -p no:cacheprovider tests/capture -q` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/ragnarok/capture/bettercam_capturer.py tests/capture/test_bettercam_timestamp.py
git commit -m "refactor(capture): injectable capture clock (arrival-time stamp seam)"
```

---

### Task 10: Single-owner mouse driver + unified fire component + reloader release

**Files:**
- Modify: `src/ragnarok/app.py`
- Modify: `src/ragnarok/gui/live_config.py`
- Test: `tests/gui/test_aim_reloader_release.py` (create)

**Interfaces:**
- Consumes: `AimReloader(loop, build_aim, commanded_buffer=None)`.
- Produces: `AimReloader` closes the outgoing controller's driver before building the new one and rolls back on build failure; `main()` builds exactly one mouse driver.

- [ ] **Step 1: Write failing test.** Create `tests/gui/test_aim_reloader_release.py`:

```python
from ragnarok.gui.live_config import AimReloader


class _Loop:
    def __init__(self): self.controller = "OLD"
    def set_aim_controller(self, c): self.controller = c


class _Cfg:
    class _A:
        enabled = True
    class _T:
        enabled = False
    aim = _A(); trigger = _T()


def test_reloader_builds_new_controller():
    loop = _Loop()
    r = AimReloader(loop, build_aim=lambda cfg, buf: "NEW")
    r.reload(_Cfg())
    assert loop.controller == "NEW"


def test_reloader_rolls_back_on_build_failure():
    loop = _Loop()
    def _boom(cfg, buf): raise RuntimeError("arduino port busy")
    r = AimReloader(loop, build_aim=_boom)
    try:
        r.reload(_Cfg())
    except RuntimeError:
        pass
    assert loop.controller == "OLD"      # stale controller kept, not clobbered with None
```

- [ ] **Step 2: Run red.** `uv run python -m pytest -p no:cacheprovider tests/gui/test_aim_reloader_release.py -q` → FAIL (`test_reloader_rolls_back_on_build_failure`: current code calls `set_aim_controller(self._build(...))` and the raise happens before rebind, so OLD is actually kept — verify. If it already passes, the failing one is the close-old behaviour below).

- [ ] **Step 3: Implement.** In `live_config.py` `AimReloader.reload`, build first (raising leaves the old controller in place), then close the previous driver after a successful swap:

```python
    def reload(self, cfg) -> None:
        if cfg.aim.enabled or cfg.trigger.enabled:
            new = self._build(cfg, self._buf)     # may raise -> old controller stays
        else:
            new = None
        prev = getattr(self._loop, "_aim", None)
        self._loop.set_aim_controller(new)
        close = getattr(prev, "close", None)      # AimController.close() releases its mouse
        if callable(close):
            try:
                close()
            except Exception:
                pass
```

Add an `AimController.close()` that releases any held trigger button and closes the mouse driver:

```python
    def close(self) -> None:
        if self._trigger is not None:
            self._trigger.release()
        m = getattr(self._mouse, "close", None)
        if callable(m):
            m()
```

- [ ] **Step 4: Single-owner mouse in `app.py`.** Refactor so `main()` builds one driver and shares it. Change `_build_fire_component`/`_build_aim_controller` to accept an injected `mouse`, and pass the same instance to `loop.set_measure_mouse`. Delete `_build_trigger_controller` (unified into `AimController`); `_build_fire_component` becomes:

```python
def _build_fire_component(cfg, commanded_buffer, mouse):
    if cfg.aim.enabled or cfg.trigger.enabled:
        return _build_aim_controller(cfg, commanded_buffer, mouse)
    return None
```

`_build_aim_controller(cfg, commanded_buffer, mouse)` uses the passed `mouse` (drop the internal `_build_mouse` call) and builds `trigger_active` in **toggle** mode:

```python
    trigger, trigger_active = _build_trigger_bot(cfg, mouse)
    # (inside _build_trigger_bot) toggle=True:
    #   trigger_active = make_aim_active(AsyncKeyStateProvider(cfg.trigger.trigger_key), toggle=True)
```

In `main()`:

```python
    mouse = _build_mouse(cfg)
    aim_controller = _build_fire_component(cfg, cmd_buffer, mouse)
    ...
    loop.set_measure_mouse(mouse)              # SAME driver, no second COM open
    ...
    app.aboutToQuit.connect(worker.stop)
    app.aboutToQuit.connect(lambda: getattr(mouse, "close", lambda: None)())
```

`AimReloader` must be constructed so its rebuild reuses the shared mouse. Simplest: have `_build_fire_component` capture the mouse via a closure passed to `AimReloader`:

```python
    aim_reloader = AimReloader(loop,
        lambda c, buf: _build_fire_component(c, buf, mouse),
        commanded_buffer=cmd_buffer)
```

`_build_trigger_bot`: change `toggle=False` → `toggle=True` for `trigger_active`.

- [ ] **Step 5: Run green + full sweep.** `uv run python -m pytest -p no:cacheprovider tests/gui tests/aim -q` → PASS. Manually confirm `app.py` imports resolve (`uv run python -c "import ragnarok.app"`).

- [ ] **Step 6: Commit.**

```bash
git add src/ragnarok/app.py src/ragnarok/gui/live_config.py src/ragnarok/aim/controller.py tests/gui/test_aim_reloader_release.py
git commit -m "fix(input): single-owner mouse driver + release on reload/exit; toggle trigger"
```

---

### Task 11: HidTransport + config routing

**Files:**
- Modify: `src/ragnarok/config/schema.py` (ArduinoConfig)
- Modify: `src/ragnarok/aim/arduino.py`
- Modify: `src/ragnarok/gui/tuning_model.py` (INPUT_FIELDS)
- Test: `tests/aim/test_arduino_transport.py` (append)

**Interfaces:**
- Consumes: `aim.protocol.encode_move/encode_button` frames (`bytes`).
- Produces: `HidTransport(vid, pid, usage_page=0xFF00)` with `open()/write(bytes)/close()`; `build_arduino_transport(cfg)` routes `transport == "hid"`.

- [ ] **Step 1: Add config fields.** In `schema.py` `ArduinoConfig`: change `transport: Literal["serial", "udp", "hid"] = "serial"`; add `vid: int = Field(default=0, ge=0, le=0xFFFF)` and `hid_pid: int = Field(default=0, ge=0, le=0xFFFF)`.

- [ ] **Step 2: Write failing test** (append to `tests/aim/test_arduino_transport.py`):

```python
def test_build_hid_transport_routes_and_guards():
    from ragnarok.aim.arduino import build_arduino_transport, HidTransport
    from ragnarok.config.schema import AppConfig, ArduinoConfig

    cfg = AppConfig(arduino=ArduinoConfig(transport="hid", vid=0x2341, hid_pid=0x0069))
    t = build_arduino_transport(cfg)
    assert isinstance(t, HidTransport)


def test_hid_transport_writes_report(monkeypatch):
    from ragnarok.aim.arduino import HidTransport

    class _FakeDev:
        def __init__(self): self.reports = []
        def write(self, data): self.reports.append(bytes(data)); return len(data)
        def close(self): pass

    dev = _FakeDev()
    t = HidTransport(vid=0x2341, pid=0x0069)
    t._dev = dev                     # inject the opened device (bypass real open)
    t.write(b"\xAA\x01\x06\x00" + b"\x00" * 6 + b"\x11")
    # HID OUTPUT reports are prefixed with a report id (0x00) and fixed length
    assert dev.reports and dev.reports[0][0] == 0x00
    assert b"\xAA\x01" in dev.reports[0]
```

- [ ] **Step 3: Run red.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_arduino_transport.py -q` → FAIL.

- [ ] **Step 4: Implement `HidTransport`** in `arduino.py` (after `UdpTransport`):

```python
_HID_REPORT_LEN = 64          # fixed OUTPUT report size (must match firmware)
_HID_USAGE_PAGE = 0xFF00      # vendor-defined


class HidTransport:  # real device I/O is box-only; framing is unit-tested
    """PC->Arduino command channel over a vendor HID OUTPUT report (driverless).

    Carries the same MAKCU frame as the serial/UDP transports, padded to a fixed
    report length and prefixed with report-id 0x00. Real device open uses hidapi
    (lazy import, box-only); tests inject ``self._dev``.
    """

    def __init__(self, vid: int, pid: int, *, usage_page: int = _HID_USAGE_PAGE,
                 report_len: int = _HID_REPORT_LEN) -> None:
        self._vid, self._pid, self._usage = vid, pid, usage_page
        self._len = report_len
        self._dev = None

    def open(self) -> None:  # pragma: no cover — box-only (real hidapi)
        import hid  # lazy: optional box-only dependency (`pip install hidapi`)
        self._dev = hid.device()
        self._dev.open(self._vid, self._pid)
        self._dev.set_nonblocking(1)

    def write(self, data: bytes) -> None:
        if len(data) > self._len:
            # chunk oversized frames across multiple reports (rare; MOVE is small)
            for i in range(0, len(data), self._len):
                self.write(data[i:i + self._len])
            return
        report = bytes([0x00]) + bytes(data) + bytes(self._len - len(data))
        self._dev.write(report)

    def close(self) -> None:
        if self._dev is not None:
            self._dev.close()
```

Update `build_arduino_transport`:

```python
def build_arduino_transport(cfg):
    a = cfg.arduino
    if a.transport == "serial":
        if not a.port:
            raise RuntimeError("arduino.port must be set for the serial transport")
        return SerialTransport(a.port, a.baud)
    if a.transport == "hid":
        if not a.vid or not a.hid_pid:
            raise RuntimeError("arduino.vid and arduino.hid_pid must be set for the hid transport")
        return HidTransport(a.vid, a.hid_pid)
    if not a.host or not a.udp_port:
        raise RuntimeError("arduino.host and arduino.udp_port must be set for the udp transport")
    return UdpTransport(a.host, a.udp_port)
```

- [ ] **Step 5: GUI fields.** In `tuning_model.py` `INPUT_FIELDS`, change the transport choices and add HID ids:

```python
    FieldSpec("arduino.transport", "Arduino transport", "choice",
              choices=("serial", "udp", "hid")),
    ...
    FieldSpec("arduino.vid", "HID vendor id", "int", 0, 0xFFFF, 1),
    FieldSpec("arduino.hid_pid", "HID product id", "int", 0, 0xFFFF, 1),
```

- [ ] **Step 6: Run green.** `uv run python -m pytest -p no:cacheprovider tests/aim/test_arduino_transport.py tests/aim/test_arduino.py -q` → PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/ragnarok/config/schema.py src/ragnarok/aim/arduino.py src/ragnarok/gui/tuning_model.py tests/aim/test_arduino_transport.py
git commit -m "feat(arduino): driverless raw-HID command transport (hidapi)"
```

---

### Task 12: Firmware — R4 + USB Host Shield passthrough (box-only)

**Files:**
- Create: `firmware/ragnarok_mouse_r4/ragnarok_mouse_r4.ino`
- Create: `firmware/ragnarok_esp32_udp/ragnarok_esp32_udp.ino`
- Create: `firmware/README.md`

**Interfaces:** must mirror `aim/protocol.py` framing/CRC exactly; MOVE payload `<hhBB>` = dx(i16) dy(i16) buttons(u8) mode(u8).

> This task is **box-only** — there is no pytest cycle. Verification is manual (flash + observe). Keep the existing `firmware/ragnarok_mouse/ragnarok_mouse.ino` (32u4/CDC) untouched for Leonardo users.

- [ ] **Step 1: Write the RA4M1 passthrough sketch.** Create `firmware/ragnarok_mouse_r4/ragnarok_mouse_r4.ino`:

```cpp
// Ragnarok mouse firmware — Arduino UNO R4 (Renesas RA4M1) + USB Host Shield.
//
// Roles:
//   HOST  (USB Host Shield / MAX3421E, SPI/ICSP): read the REAL mouse's HID reports.
//   DEVICE(native USB-C): present ONE HID mouse to the PC = passthrough + injected aim.
//   CMD   : receive MAKCU frames (aim deltas / clicks) from the PC via
//           (a) a vendor HID OUTPUT report, and (b) Serial1 (the ESP32 WiFi link).
//
// Frame MUST match src/ragnarok/aim/protocol.py EXACTLY:
//   [0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]  CRC8 poly 0x07 init 0x00 over CMD+LEN+PAYLOAD
//   CMD_MOVE 0x01 <hhBB>=dx,dy,buttons,mode  CMD_BUTTON 0x02 <B>=mask
//   CMD_CONFIG 0x03 opaque  CMD_DIAG 0x04 <B>=seq -> echo CMD_DIAG <I>=micros()
//
// Libraries: felis/USB_Host_Shield_2.0 (R4-compatible), Mouse.h (native HID).
// NOTE: on the R4 WiFi the native HID takes the USB-C lines (D40 mux); upload via
// the normal path then it enumerates as the HID mouse. See firmware/README.md.

#include <SPI.h>
#include <usbhid.h>
#include <hiduniversal.h>
#include <usbhub.h>
#include <Mouse.h>

static const uint8_t START = 0xAA, CMD_MOVE = 0x01, CMD_BUTTON = 0x02,
                     CMD_CONFIG = 0x03, CMD_DIAG = 0x04;

static uint8_t crc8_step(uint8_t crc, uint8_t b) {
  crc ^= b;
  for (uint8_t i = 0; i < 8; i++)
    crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  return crc;
}
static uint8_t crc8_buf(const uint8_t* d, uint16_t n) {
  uint8_t c = 0; for (uint16_t i = 0; i < n; i++) c = crc8_step(c, d[i]); return c;
}

// ---- injected-command accumulator (from PC over HID/Serial1) ----
static volatile int32_t inj_dx = 0, inj_dy = 0;
static volatile uint8_t inj_buttons = 0;

static void setButtons(uint8_t mask) {
  (mask & 0x01) ? Mouse.press(MOUSE_LEFT)   : Mouse.release(MOUSE_LEFT);
  (mask & 0x02) ? Mouse.press(MOUSE_RIGHT)  : Mouse.release(MOUSE_RIGHT);
  (mask & 0x04) ? Mouse.press(MOUSE_MIDDLE) : Mouse.release(MOUSE_MIDDLE);
}
static void emitMove(int32_t dx, int32_t dy) {         // chunk to int8 HID range
  while (dx != 0 || dy != 0) {
    int8_t sx = dx > 127 ? 127 : (dx < -127 ? -127 : (int8_t)dx);
    int8_t sy = dy > 127 ? 127 : (dy < -127 ? -127 : (int8_t)dy);
    Mouse.move(sx, sy, 0);
    dx -= sx; dy -= sy;
  }
}
static void sendDiagEcho() {
  uint32_t us = micros();
  uint8_t body[7] = { CMD_DIAG, 0x04, 0x00,
                      (uint8_t)us, (uint8_t)(us >> 8), (uint8_t)(us >> 16), (uint8_t)(us >> 24) };
  Serial.write(START); Serial.write(body, sizeof(body)); Serial.write(crc8_buf(body, sizeof(body)));
}

static void handleFrame(uint8_t cmd, const uint8_t* buf, uint16_t len) {
  switch (cmd) {
    case CMD_MOVE:
      if (len >= 6) {
        inj_dx += (int16_t)(buf[0] | (buf[1] << 8));
        inj_dy += (int16_t)(buf[2] | (buf[3] << 8));
        inj_buttons = buf[4];
      }
      break;
    case CMD_BUTTON: if (len >= 1) inj_buttons = buf[0]; break;
    case CMD_DIAG:   sendDiagEcho(); break;
    default: break;
  }
}

// ---- byte-at-a-time MAKCU parser (reused for Serial1 + HID OUTPUT bytes) ----
struct Parser {
  uint8_t state = 0, cmd = 0, lenLo = 0, rc = 0;
  uint16_t len = 0, idx = 0; uint8_t buf[264];
  void feed(uint8_t b) {
    switch (state) {
      case 0: if (b == START) state = 1; break;
      case 1: cmd = b; rc = crc8_step(0, b); state = 2; break;
      case 2: lenLo = b; rc = crc8_step(rc, b); state = 3; break;
      case 3: len = (uint16_t)lenLo | ((uint16_t)b << 8); rc = crc8_step(rc, b);
              idx = 0; state = (len > sizeof(buf)) ? 0 : (len ? 4 : 5); break;
      case 4: buf[idx++] = b; rc = crc8_step(rc, b); if (idx >= len) state = 5; break;
      case 5: if (rc == b) handleFrame(cmd, buf, len); state = 0; break;
    }
  }
};
static Parser serialParser;   // Serial1 (ESP32 link)
static Parser hidParser;      // vendor HID OUTPUT report bytes

// ---- USB Host Shield: real-mouse reader -> passthrough deltas ----
USB Usb;
class RealMouse : public HIDUniversal {
public:
  RealMouse(USB* p) : HIDUniversal(p) {}
protected:
  void ParseHIDData(USBHID*, bool, uint8_t, uint16_t len, uint8_t* buf) {
    // Boot-mouse report: [buttons][dx][dy][wheel]; 16-bit-capable mice vary — adjust
    // to your device's report layout (see firmware/README.md).
    if (len >= 3) {
      int8_t dx = (int8_t)buf[1], dy = (int8_t)buf[2];
      Mouse.move(dx, dy, (len >= 4) ? (int8_t)buf[3] : 0);   // passthrough real motion
      setButtons((buf[0] & 0x07) | inj_buttons);              // real+injected buttons
    }
  }
};
RealMouse realMouse(&Usb);

void setup() {
  Serial.begin(115200);     // native USB CDC (optional debug) — HID is separate
  Serial1.begin(921600);    // ESP32-S3 link (WiFi command path)
  Mouse.begin();
  if (Usb.Init() == -1) { /* host shield init failed — check wiring */ }
}

void loop() {
  Usb.Task();                               // pumps the real mouse -> passthrough
  while (Serial1.available() > 0) serialParser.feed((uint8_t)Serial1.read());
  // TODO(box): feed hidParser from the vendor HID OUTPUT report callback once the
  // composite descriptor is added (see firmware/README.md "Vendor HID channel").
  int32_t dx = inj_dx, dy = inj_dy; inj_dx = 0; inj_dy = 0;   // drain injected aim
  if (dx || dy) emitMove(dx, dy);
}
```

- [ ] **Step 2: Write the ESP32-S3 UDP firmware.** Create `firmware/ragnarok_esp32_udp/ragnarok_esp32_udp.ino`:

```cpp
// Ragnarok WiFi command bridge — ESP32-S3 (UNO R4 WiFi radio module).
// Replaces the stock USB-serial bridge: receives MAKCU frames over UDP and
// forwards the raw bytes to the RA4M1 over the internal UART. See firmware/README.md
// for the espflash upload procedure (this OVERWRITES the stock bridge firmware).
#include <WiFi.h>
#include <WiFiUdp.h>

const char* SSID = "YOUR_SSID";
const char* PASS = "YOUR_PASS";
const uint16_t CMD_PORT = 9999;

WiFiUDP udp;
uint8_t pkt[512];

void setup() {
  Serial1.begin(921600);                 // UART to the RA4M1
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) delay(100);
  udp.begin(CMD_PORT);
}

void loop() {
  int n = udp.parsePacket();
  if (n > 0) {
    int len = udp.read(pkt, sizeof(pkt));
    if (len > 0) Serial1.write(pkt, len);   // forward MAKCU frame(s) verbatim
  }
}
```

- [ ] **Step 3: Write `firmware/README.md`** documenting: (a) the three-role topology diagram, (b) required libraries + versions, (c) the D40 native-HID mux note and double-tap-reset re-flash workflow, (d) the vendor-HID descriptor edit needed for the `HidTransport` command channel (which core file, and that until it's added the command channel is Serial1/UDP only), (e) the espflash steps to replace the ESP32 bridge, (f) that MOVE deltas are chunked to int8 and that a 16-bit report descriptor is a future refinement, (g) how to set `arduino.vid`/`arduino.hid_pid` (or serial port / UDP host+port) in the Ragnarok config to match.

- [ ] **Step 4: Manual verification (box-only, user).** Document as the acceptance check (no CI): flash R4 sketch → plug real mouse into the shield → confirm the OS sees one mouse and it moves 1:1 → run Ragnarok with `input.mouse_driver=arduino`, `arduino.transport=serial` (or `hid`/`udp`) → enable trigger → confirm injected aim/clicks reach the game → `scripts/measure_hil.py` returns a round-trip.

- [ ] **Step 5: Commit.**

```bash
git add firmware/ragnarok_mouse_r4/ firmware/ragnarok_esp32_udp/ firmware/README.md
git commit -m "feat(firmware): R4 + USB Host Shield passthrough + ESP32 UDP bridge (box-only)"
```

---

## Self-Review

**Spec coverage:**
- §3.1 aimer commit/settle → Task 1 ✓
- §3.2 snappy defaults → Task 2 ✓
- §3.2b toggle activation → Tasks 2 (aim), 5 (trigger), 10 (build toggle) ✓
- §3.3 capture-time timestamp → Task 9 (CI seam; DXGI box-only, documented) ✓
- §3.4 latency latch → Task 7 ✓
- §3.5 HidTransport + single-owner driver + firmware → Tasks 10, 11, 12 ✓
- §3.6 TOCTOU + shutdown → Task 8 ✓
- §3.7 trigger rework → Tasks 3, 4, 6, 10 ✓

**Placeholder scan:** the only `TODO` is the firmware `hidParser` feed, which is legitimately box-only (needs the composite-descriptor edit) and documented in Step 3's README item — not a plan placeholder. All Python steps carry complete code.

**Type consistency:** `commit`/`settle_px` param names match across `aimers.py`, `build_aimer`, and `AIM_FIELDS`. `max_occlusion_frames` matches across `TriggerBot`, `TriggerConfig`, and `TRIGGER_FIELDS`. `aim_on`/`trigger_on` match across `AimController`, `TelemetrySnapshot`, and `loop.py`. `HidTransport(vid, pid, ...)` matches `build_arduino_transport`. `fire_target_id` is introduced and used only within `AimController`.

**Note on Task 4 ↔ Task 10 coupling:** Task 4 removes `trigger/controller.py`; Task 10 removes its last `app.py` usage. If executed out of order, run the import check in Task 10 Step 5 before committing Task 4. Tasks are otherwise independently testable.
