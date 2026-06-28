# Ragnarok Phase 3 — Aim Core Implementation Plan

> Implement task-by-task, TDD, one commit per task. Executed via a Workflow (ultracode); equally runnable inline.

**Goal:** First working aim — holding an aim key moves the crosshair toward the selected ENEMY track: FOV gate → sticky target selection → per-track IMM lead → Flick/Feedback aimer → SendInput mouse driver.

**Architecture:** A new `src/ragnarok/aim/` package. `AimController.update(tracks, t_capture_ns)` runs **after** classify each tick, gated by `is_aim_active()`. It selects an ENEMY target (dual-radius sticky FOV + dwell + switch-margin), updates a per-track filterpy IMM (CV+CA) and reads a lead aim point, asks the chosen `Aimer` for a pixel delta, converts px→mouse-counts, and calls `MouseDriver.move_relative`. Everything runs in **pixel space** (identity ego-motion); a deg↔px seam isolates the Phase-4 swap to world-angular. Every collaborator is constructor-injected → fully CI-safe with `NullMouseDriver` + fake key/clock providers.

**Tech Stack:** numpy, scipy, **filterpy** (new dep, pure-Python), ctypes (Windows SendInput/GetAsyncKeyState — lazily bound so import stays cross-platform), pydantic, pytest.

## Global Constraints

- **Pixel space, identity ego-motion** this phase; world-angular (spec §6.4) deferred to Phase 4. Keep deg↔px math in one place.
- **ENEMY-only targeting** (Team.ENEMY): UNKNOWN/TEAMMATE are never selected — assert in the selector (Phase 2 safety contract).
- **CI-safe**: no real mouse/keyboard/GPU/display in unit tests. `ctypes.WinDLL`/`windll` lookups happen **lazily inside `__init__`**, never at import, so the package imports on CI. ABCs + `Null*`/`Fake*` fakes are the test seams; injected clock for timing.
- **Relative mouse only**: `MOUSEEVENTF_MOVE` without `MOUSEEVENTF_ABSOLUTE`; `dwExtraInfo` typed `wintypes.ULONG_PTR`; `cbSize = sizeof(INPUT)`; set `argtypes`/`restype`. Sub-pixel **fractional accumulator** in the driver so small deltas aren't truncated to 0.
- **dt from `frame.t_capture_ns`** (not wall clock), clamped to `[1e-3, 0.1]` s.
- **Backward compatible**: `AppConfig` gains `aim: AimConfig = AimConfig()`; `WorkerLoop` gains `aim_controller=None`. All 97 existing tests must still pass.
- **Aimer contract**: `step(crosshair, target_point, dt) -> (dx_px, dy_px)`, clamped, never overshoot. FlickAimer **latches** target on first call after `reset()`.
- **Deferred to Phase 4**: Hybrid/Predictive aimers, WindMouse shaping, `Kff·v` feed-forward, recoil, trigger bot, Interception/Arduino drivers, world-angular + real feed-forward GMC, adaptive lead.
- TDD, DRY, YAGNI, one commit per task.

## File Structure
```
src/ragnarok/config/schema.py      # MODIFY: add AimConfig, nest in AppConfig
src/ragnarok/aim/__init__.py       # new
src/ragnarok/aim/mouse.py          # MouseDriver ABC, NullMouseDriver, SendInputMouseDriver
src/ragnarok/aim/keys.py           # AimKeyProvider ABC, FakeKeyProvider, AsyncKeyStateProvider, make_aim_active
src/ragnarok/aim/fov.py            # focal_length_px, fov_deg_to_radius_px, aim_point, distance helpers
src/ragnarok/aim/select.py         # select_target (pure) + TargetSelector (sticky/dwell/margin)
src/ragnarok/aim/aimers.py         # Aimer ABC, NullAimer, FlickAimer, FeedbackAimer
src/ragnarok/aim/imm.py            # TrackIMM (filterpy CV+CA) + IMMManager
src/ragnarok/aim/controller.py     # AimController
src/ragnarok/worker/loop.py        # MODIFY: aim_controller kwarg + 'aim' stage
src/ragnarok/app.py                # MODIFY: wire real driver/keys/controller (Windows)
pyproject.toml                     # MODIFY: add filterpy>=1.4.5
tests/aim/...                      # mirrors
```

---

### Task 1: AimConfig
**Files:** Modify `config/schema.py`; Test `tests/config/test_aim_config.py`.
**Produces:** frozen `AimConfig` nested as `AppConfig.aim` (default instance → backward compatible).

```python
class AimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    aim_key: str = "VK_RBUTTON"
    toggle: bool = False                       # False = hold-to-aim
    hfov_deg: float = Field(default=90.0, gt=0.0, le=180.0)
    screen_width_px: int = Field(default=1920, ge=320, le=7680)
    aim_fov_deg: float = Field(default=5.0, gt=0.0, le=179.0)      # acquire (inner)
    retain_fov_deg: float = Field(default=8.0, gt=0.0, le=179.0)   # keep (outer) > inner
    dwell_ms: float = Field(default=100.0, ge=0.0, le=2000.0)
    switch_margin: float = Field(default=0.20, ge=0.0, lt=1.0)
    aimer: Literal["flick", "feedback"] = "feedback"
    kp: float = Field(default=0.35, gt=0.0, le=2.0)
    max_step_px: float = Field(default=60.0, gt=0.0)
    flick_speed_px_s: float = Field(default=4000.0, gt=0.0)
    ema_alpha: float = Field(default=0.5, gt=0.0, le=1.0)
    aim_point: Literal["head", "body"] = "head"
    head_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    sensitivity: float = Field(default=0.022, gt=0.0)              # deg per mouse count
    lead_ms: float = Field(default=40.0, ge=0.0, le=500.0)
```
Add `aim: AimConfig = AimConfig()` to `AppConfig`. **Tests:** defaults; frozen; `AppConfig().aim.aimer == "feedback"`; bad `aimer` rejected; existing `AppConfig()` still valid. Commit `feat(config): add AimConfig`.

---

### Task 2: Mouse driver
**Files:** `aim/__init__.py`, `aim/mouse.py`; Test `tests/aim/test_mouse.py`.
**Produces:** `MouseDriver` ABC `move_relative(dx: float, dy: float)`, `set_button(button, down)` (Phase-3 no-op), `connect()/close()`; `NullMouseDriver` (records `.moves`); `SendInputMouseDriver(send=None, max_px_per_tick=32767)`.

Use the prep's robust ctypes (full code in prep output): `MOUSEINPUT/INPUT` with `wintypes.ULONG_PTR`, anonymous union, `_make_real_send()` setting `argtypes/restype` and `cbSize=sizeof(INPUT)`. The driver holds a **float remainder accumulator**:
```python
def move_relative(self, dx, dy):
    self._rx += dx; self._ry += dy
    ix = int(self._rx); iy = int(self._ry)         # toward zero
    self._rx -= ix; self._ry -= iy
    if ix == 0 and iy == 0: return
    ix = max(-self._max, min(self._max, ix)); iy = max(-self._max, min(self._max, iy))
    self._send(ix, iy, MOUSEEVENTF_MOVE)
```
`connect()` binds `_make_real_send()` (lazy WinDLL) and zeros the accumulator; `NullMouseDriver._send` records `(ix,iy,flags)`.
**Tests (inject a fake `send`):** struct/flags correct (`MOUSEEVENTF_MOVE`, no ABSOLUTE); **accumulator** — repeated `move_relative(0.4, 0)` emits a `1` only every ~3 frames and total ≈ commanded; max clamp; `(0,0)` emits nothing; `NullMouseDriver.moves` records. Commit `feat(aim): SendInput mouse driver with sub-pixel accumulator`.

---

### Task 3: Aim-key provider
**Files:** `aim/keys.py`; Test `tests/aim/test_keys.py`.
**Produces:** `AimKeyProvider` ABC `is_down()`; `FakeKeyProvider(down=False)`; `AsyncKeyStateProvider(key_name)` (lazy WinDLL; `GetAsyncKeyState(vk) & 0x8000`); `make_aim_active(provider, *, toggle)` → closure (hold = raw; toggle flips on rising edge); `VK` name→code map.
**Tests:** hold mode mirrors `FakeKeyProvider.down`; toggle flips only on rising edge (down→down stays); `VK["VK_RBUTTON"]==0x02`. (`AsyncKeyStateProvider` not constructed in CI.) Commit `feat(aim): injectable aim-key provider (hold/toggle)`.

---

### Task 4: FOV + target selection
**Files:** `aim/fov.py`, `aim/select.py`; Tests `tests/aim/test_fov.py`, `test_select.py`.
**Produces:** `focal_length_px(hfov_deg, screen_w)`, `fov_deg_to_radius_px(fov_deg, hfov_deg, screen_w)`, `crosshair_for_roi(w,h)`, `aim_point(track, head_frac, mode)`, `dist_to`; pure `select_target(tracks, crosshair, fov_px, *, head_frac, current_target_id=None, retain_fov_px=None, switch_margin=0.0)`; stateful `TargetSelector(fov_inner_px, fov_outer_px, dwell_ms, switch_margin, head_frac, clock=now_ns)` with `select(tracks, cx, cy)` and `reset()`.

Use the prep code (fov.py math; select.py scoring + dual-radius sticky + dwell + switch-margin; **ENEMY-only**; tie-break by track_id). `f = (screen_w/2)/tan(hfov/2)`, `radius = f·tan(fov/2)`.
**Tests:** `focal_length_px(90,1920)==960`; radius monotonic; nearest-enemy selection; TEAMMATE/UNKNOWN never chosen; switch-margin keeps lock; dwell-then-switch (FakeClock); lock resets on target death/FOV-exit. Commit `feat(aim): FOV cone and sticky target selection`.

---

### Task 5: Aimers
**Files:** `aim/aimers.py`; Test `tests/aim/test_aimers.py`.
**Produces:** `Aimer` ABC `step(crosshair, target_point, dt)->(dx,dy)` + `reset()`; `NullAimer`; `FlickAimer(flick_speed_px_s)` (latch target on first call after reset, glide at speed, clamp to remaining distance — no overshoot, no re-acquire until `reset()`); `FeedbackAimer(kp, max_step_px, ema_alpha)` (P-controller on live error, EMA-smoothed, clamped). `Kff` param present but unused (Phase-4 hook).
Use prep aimers.py code.
**Tests:** flick clamps to remaining distance (no overshoot) and stays latched until `reset()`; flick re-acquires after `reset()`; feedback `dx≈Kp·error`; feedback clamps to `max_step_px`; EMA smooths across frames. Commit `feat(aim): Flick and Feedback (P-controller) aimers`.

---

### Task 6: IMM motion filter
**Files:** `aim/imm.py`; Modify `pyproject.toml` (+`filterpy>=1.4.5`); Test `tests/aim/test_imm.py`.
**Produces:** `TrackIMM(x0, y0)` (filterpy `IMMEstimator` of CV+CA 6-state `[x,vx,ax,y,vy,ay]`, `mu=[0.5,0.5]`, `M=[[0.95,0.05],[0.05,0.95]]`); `update(x, y, dt)` rebuilds F (and Q via `Q_discrete_white_noise(dim=3, dt, var, block_size=2, order_by_dim=True)`; CV var small, CA var large), `predict()` then `update(z)`; `position()/velocity()`; `lead(t_lead)` = pos + v·t + ½a·t². `IMMManager` keyed by track_id: `update(tid, x, y, dt)`, `lead(tid, t_lead)`, `prune(live_ids)`; inflate P + reset mu on re-acquire.
Use the prep IMM code (reconciled: per-frame F/Q from dt; lead with accel term).
**Tests (synthetic, injected dt):** constant-velocity sequence → position RMSE small + velocity ≈ injected; a juke → CA mode prob rises and estimate re-tracks within N frames; `IMMManager.prune` drops dead ids; install `filterpy` then run. Commit `feat(aim): per-track IMM (CV+CA) lead via filterpy`.

---

### Task 7: AimController
**Files:** `aim/controller.py`; Test `tests/aim/test_controller.py`.
**Produces:** `AimController(cfg: AimConfig, *, selector, imm_manager, aimer, mouse, is_aim_active, roi_size, clock=now_ns)`; `update(tracks, t_capture_ns)`; `.target_id`.
Logic: prune IMM to live ids; if not `(cfg.enabled and is_aim_active())` → `aimer.reset()`, `selector.reset()`, clear `target_id`, reset dt baseline, return (no mouse calls). Else: `target = selector.select(...)`; on target_id change call `aimer.reset()`; compute `aim_point`; `imm_manager.update(tid, ax, ay, dt)`; `lead_pt = imm_manager.lead(tid, lead_s)`; `dpx,dpy = aimer.step((cx,cy), lead_pt, dt)`; `counts = (dpx,dpy) * (deg_per_px / sensitivity)`; `mouse.move_relative(*counts)`. `deg_per_px = aim... ` use `hfov_deg/screen_width_px` (deg per screen px). dt from `t_capture_ns` deltas, clamped.
**Tests (NullMouseDriver + FakeKeyProvider/clock):** inactive → `mouse.moves == []`; active + enemy right-of-crosshair → commanded `dx>0` over a few ticks; UNKNOWN/TEAMMATE → `target_id is None`, no move; target switch resets the flick latch; disengage resets aimer/selector. Commit `feat(aim): AimController orchestration`.

---

### Task 8: Worker integration + app wiring
**Files:** Modify `worker/loop.py`, `app.py`; Test extend `tests/worker/test_loop.py`.
**Produces:** `WorkerLoop(..., aim_controller=None)`; in `tick()` after classify: `if self._aim is not None: self._aim.update(tracks, frame.t_capture_ns)`; add profiler `"aim"` stage; loop time spans through aim. `app.py` (Windows): build `SendInputMouseDriver`, `AsyncKeyStateProvider`+`make_aim_active`, selector/imm/aimer from `cfg.aim`, `AimController`, pass to `WorkerLoop` — only when `cfg.aim.enabled`.
**Tests:** existing Phase-1/2 worker tests pass unchanged (default `aim_controller=None`); injecting a fake aim_controller → its `update` is called with the tracks; `"aim"` stage present when controller set. Then **full suite green** (`QT_QPA_PLATFORM=offscreen python -m pytest -q`). Commit `feat(worker,app): wire AimController into the loop`.

---

## Self-Review
- **Spec §17.3 coverage:** FOV (T4) ✓, smart selection + hysteresis (T4) ✓, IMM (T6) ✓, Flick + P-controller (T5) ✓, SendInput driver (T2) ✓, first working aim wired + key-gated (T7/T8) ✓. World-angular/Predictive/Hybrid/WindMouse/recoil/trigger correctly deferred to Phase 4.
- **Placeholders:** none — concrete code or exact prep-grounded specs per step.
- **Type consistency:** `MouseDriver.move_relative(dx,dy)`, `Aimer.step(crosshair, target_point, dt)`, `IMMManager.update(tid,x,y,dt)/lead(tid,t)/prune`, `TargetSelector.select(tracks,cx,cy)`, `AimController(cfg,*,selector,imm_manager,aimer,mouse,is_aim_active,roi_size)`, `AppConfig.aim`, `WorkerLoop(...,aim_controller=)` — consistent across T7/T8.
- **CI-safe:** ctypes bound lazily; all side-effecting collaborators injected; tests use Null/Fake fakes + injected clock. Real SendInput/GetAsyncKeyState/end-to-end aim are the user-box manual smokes.
