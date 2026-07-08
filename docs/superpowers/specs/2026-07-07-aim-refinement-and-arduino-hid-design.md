# Ragnarok — Aim Refinement + Arduino HID/WiFi (Approach A) Design

**Date:** 2026-07-07
**Status:** Approved (Approach A), pre-implementation
**Predecessor:** `docs/superpowers/specs/2026-06-26-ragnarok-design.md` (rev 4)

## 1. Motivation

Live-play feedback: `feedback` aiming "works well enough" but the `flick`,
`hybrid`, and `predictive` aimers **overshoot**, and the aim as a whole is not as
**refined/snappy** as it should be — specifically **wobble/creep on arrival** and
a **jittery/unsteady hold**. The **trigger bot doesn't really work** and must fire
independently of auto-aim (shoot when the crosshair is on an enemy). Auto-aim and
the trigger should each be an **independent on/off toggle** on a non-obtrusive
hotkey, with **hold-to-aim removed**. Separately, the Arduino output path must be
reachable over **USB HID** and over **socket/WiFi**, targeting the user's real
hardware: an **Arduino UNO R4 WiFi + USB Host Shield** (not the 32u4 the current
firmware assumes).

This is Approach A: fix the overshoot and jitter at their sources, make the
calibration that prevents overshoot actually apply, fix the wiring bugs that
block Arduino use, and rebuild the firmware around the correct Host-Shield
passthrough topology. It deliberately does **not** add a decoupled high-rate aim
thread (Approach B) — the user did not report "slow to reach target," so the
payoff there does not match the complaint.

## 2. Root-cause analysis (verified against current code)

### 2.1 Why open-loop aimers overshoot, feedback doesn't

`AimController.update` (`aim/controller.py`) computes a Smith-predictor crosshair
`(chx, chy)` — the ROI centre advanced by counts commanded within
`aim.deadtime_ms` — then calls `aimer.step((chx,chy), lead_pt, dt, ...)`.

- `FeedbackAimer` commands only `kp·ē` per tick (`kp=0.35`) plus a creep zone and
  sign-flip anti-windup. Residual error from imperfect dead-time/calibration is
  corrected **gradually** — the loop is robust to calibration error.
- `PredictiveAimer.step` (`aimers.py:321`) commands the **full** positional error
  to the led point every tick and clamps to `max_step_px` **only, not remaining
  distance** — the sole aimer that doesn't clamp to remaining. During approach the
  full error is re-issued each tick before the move is visible → stacking →
  overshoot.
- `HybridAimer` close leg (`aimers.py:280`) **snaps the full remaining error**
  within `flick_dist_px` (20 px) every tick → terminal wobble.
- `FlickAimer` glides at a fixed `flick_speed_px_s` (4000) with **no proportional
  damping**; a mis-estimated dead-time keeps it issuing full-speed steps for
  motion already in flight.

**Conclusion:** the three non-feedback aimers are open-loop — correct only if the
Smith predictor perfectly cancels in-flight motion. It cannot, because (a) the
latency calibration almost never applies (§2.3), (b) the Smith window is anchored
at `now` rather than the frame time (under-covers processing latency), and
(c) `sensitivity`/`deg_per_count` are usually uncalibrated.

### 2.2 Why the hold is jittery / arrival wobbles

- **No settle deadzone anywhere.** The sub-pixel accumulator (`mouse._FracAccumulator`)
  emits ±1 px moves indefinitely while chasing a noisy target point.
- **Schema defaults reintroduce lead jitter.** `AimConfig` defaults are
  `adaptive_lead=True`, `lead_ms=40` (`schema.py:56,62`). Phase 9A set these to
  `0`/off for snappiness, but only in the user's local `config.toml`; a fresh
  config or profile still applies a 40 ms + frame-age lead to a **noisy IMM
  velocity estimate**, so even a stationary target's lead point jitters
  frame-to-frame and the crosshair chases it.
- **Consume-time frame timestamps.** `BetterCamCapturer.grab` stamps
  `t_capture_ns = now_ns()` at consume time, not frame arrival
  (`bettercam_capturer.py:29`), injecting ±ms of `dt` jitter into IMM velocity,
  which amplifies the lead wobble.

### 2.3 The anti-overshoot calibration is silently broken (confirmed bug)

`WorkerLoop.tick` publishes `latency_ms=self._measure_ms` then immediately sets
`self._measure_ms=None` (`loop.py:134-136`), so the measured latency lives in
exactly one snapshot (~7 ms at 144 Hz). `CountsCalibratePanel._check_latency`
polls the publisher every 200 ms (`counts_panel.py:143`), catching it ~3% of the
time. So "Measure latency" almost never writes `aim.deadtime_ms` /
`tracking.tau_render_s`; the Smith predictor runs on the 40 ms default and
in-flight commands older than that are re-corrected → overshoot.

### 2.4 Arduino wiring bugs that block device use today (confirmed)

1. `app.py:240` builds a **second** mouse driver for the latency-measure feature.
   With `input.mouse_driver="arduino"` + `arduino.transport="serial"`, this
   double-opens the exclusive COM port → `SerialException` → **main() crashes on
   startup**.
2. `AimReloader.reload` (`live_config.py:20`) builds the new controller (opening a
   fresh serial port) **before** releasing the old one, and no code path ever
   `close()`s a driver. On Arduino, **every live tuning edit** raises at build,
   is swallowed as a warning, and aim keeps running the stale controller.
3. Firmware targets a 32u4 over USB-CDC only; the user's board is a UNO R4 WiFi
   with a USB Host Shield and there is **no WiFi listener and no passthrough** on
   the device at all.

### 2.5 Correctness bugs found in passing

- Detector-read TOCTOU: `loop.py:103` does `hasattr(self._det, "observe_lock")`
  then `self._det.observe_lock(...)` as two reads; a GUI-thread `set_detector`
  between them raises `AttributeError` and kills the worker thread.
- `BetterCamCapturer.grab` → bettercam `get_latest_frame()` waits with **no
  timeout**; a static screen blocks `tick()` indefinitely, so `stop_event` is
  never checked → 2 s hang and unclean shutdown, and aim/trigger freeze on static
  scenes.

### 2.6 Trigger bot: coupled to aim, and inert by default (verified)

The standalone `TriggerController` logic is correct in isolation (its tests pass),
but the way it is gated makes it "not really work":

- **Chained to the aim key.** With `aim.enabled=True` (default), the trigger runs
  *inside* `AimController.update`, which early-returns unless the aim key is held
  (`controller.py:84`). So the trigger only fires *while auto-aim is engaged* —
  never independently. The separate `TriggerController` (aim-off path) exists but
  is easy to miss and shares the problems below.
- **Default activation key is the fire button.** `trigger.trigger_key`
  defaults to `VK_LBUTTON` and `button` to `left`; holding left (which already
  fires) to "activate" a bot that also clicks left is a no-op to the eye.
- **Hard `Team.ENEMY` gate.** If friend/foe is on but can't tag a target (tiny /
  odd-coloured boxes), tracks stay `UNKNOWN` and it silently never fires. In a
  single-player sandbox everything under the crosshair is a valid target.
- **Reaction delay resets on 1-frame tracking gaps.** `activation_delay_ms=80`
  with `occluded = time_since_update > 0` (`bot.py`) means any single missed
  detection frame resets `_eligible_since`, so on flickery tracks the delay never
  accumulates → feels dead.

## 3. Design

### 3.1 Aimer unification — commit fraction + terminal settle (fixes 2.1, 2.2)

Add two shared, configurable behaviours applied inside the aimer layer, so each
aimer keeps its character but none is 100% open-loop:

- **Commit fraction** (`aim.commit`, default `0.85`, range `(0, 1]`): open-loop
  aimers issue `commit × step` rather than the full computed step. A commit < 1
  makes flick/predictive/hybrid self-damp against imperfect dead-time and
  calibration exactly the way `kp < 1` does for feedback (this is the
  sunone/NeuralBot "full-error × 0.85" insight already noted in project memory).
  `commit = 1.0` reproduces today's behaviour byte-for-byte.
- **Settle deadzone** (`aim.settle_px`, default `2.0`, range `[0, …)`): when the
  post-Smith error magnitude is `≤ settle_px`, the aimer commands `(0, 0)`. This
  is the single fix for both the arrival wobble and the jittery hold — it stops
  the sub-pixel accumulator chasing sub-pixel noise. `settle_px = 0` disables it
  (current behaviour).

Per-aimer changes (all operate on the controller-supplied `(crosshair, target)`,
so the Smith predictor still feeds them the advanced crosshair):

| Aimer | Change |
|---|---|
| `FlickAimer` | apply settle deadzone; glide unchanged otherwise (already clamps to remaining) |
| `HybridAimer` | close-leg snap becomes `commit ×` remaining; settle deadzone; far leg unchanged |
| `PredictiveAimer` | **clamp to `min(max_step, remaining)`** (was `max_step` only); apply `commit ×`; settle deadzone |
| `FeedbackAimer` | settle deadzone only; **ignores `commit`** (its damping is `kp`); `creep_px` and all PID math unchanged, so it stays byte-identical at today's defaults |

**Note on feedback back-compat:** `commit` defaults to `0.85`, which would change
`FeedbackAimer` output. To preserve the §15 CI step-response regression (locks
no-overshoot on P and PID) and the memory's "byte-identical feedback" guarantee,
`commit` is applied **only to the open-loop aimers** (`flick`/`hybrid`/`predictive`);
`FeedbackAimer` ignores `commit` (its damping already comes from `kp`). The settle
deadzone applies to all four (it only zeroes sub-`settle_px` motion, which cannot
cause overshoot and is compatible with the regression's overshoot metric). This
keeps feedback exactly as tuned while fixing the three that overshoot.

### 3.2 Snappy, steady schema defaults (fixes 2.2)

Promote the Phase-9A snappy tuning to the actual `AimConfig` schema defaults so a
fresh config/profile is steady out of the box:

- `adaptive_lead`: `True` → `False`
- `lead_ms`: `40.0` → `0.0`
- add `commit`, `settle_px` (defaults above)

`dwell_ms`/`switch_margin` are left at their current values (the user did not
report target-switch lag). Existing saved `config.toml` values are unaffected
(defaults only fill unset fields).

### 3.2b Activation model — independent toggles, no hold-to-aim (user request)

Auto-aim and the trigger bot each become a **runtime toggle** flipped by its own
dedicated, non-obtrusive hotkey; **hold-to-aim is removed** as the shipped model.

- **Auto-aim toggle.** `aim.toggle` defaults to `True` (rising-edge toggle via the
  existing `make_aim_active(..., toggle=True)` closure). `aim.aim_key` becomes the
  *toggle* key; its default moves off `VK_RBUTTON` (ADS in most games) to a
  non-obtrusive default (mouse side button `VK_XBUTTON2`). Pressing it once
  engages continuous auto-aim; pressing again disengages. The `toggle=False` (hold)
  path remains in code as an escape hatch but is no longer the default.
- **Trigger toggle.** The trigger gets its own toggle key `trigger.trigger_key`,
  defaulting to the other side button (`VK_XBUTTON1`), and is evaluated in
  **toggle** mode (was hard-coded hold). When toggled on, it fires whenever the
  crosshair is inside an enemy hitbox (§3.7). Its `button` (what it clicks) stays
  `left`; the *toggle* key is now distinct from the clicked button, fixing the
  self-conflict.
- **State visibility.** Because toggles have no held-key feedback, the current
  `AIM: ON/OFF` and `TRIGGER: ON/OFF` state is surfaced in the overlay HUD and the
  Dashboard so the user always knows what is armed. The controller exposes the
  live toggle state for the telemetry snapshot to publish.
- Keybinds tab exposes both toggle keys; both are collision-checked against the
  Calibrate hotkeys (`VK_HOME`/`VK_END`).

### 3.3 Capture-time timestamps (reduces 2.2 at the source)

Stamp `t_capture_ns` at frame **arrival** instead of consume time. Preferred:
read `DXGI_OUTDUPL_FRAME_INFO.LastPresentTime` (already populated in bettercam's
duplicator) and carry it through `Frame.t_capture_ns`. Fallback if that field is
not cleanly reachable: a thin capture thread that stamps `now_ns()` the instant
bettercam signals a new frame. This removes the 0–1-frame hidden age and the
`dt` jitter feeding IMM velocity. The `Frame` dataclass already carries
`t_capture_ns`; only the producer changes, so downstream is untouched.

### 3.4 Calibration reliability — latch the latency result (fixes 2.3)

Latch `latency_ms` in the published snapshot until the next measurement request,
instead of clearing it after one publish. `CountsCalibratePanel._check_latency`
then reliably catches it. Concretely: `WorkerLoop` keeps `self._measure_ms` set
after publishing and only clears it when a **new** `request_latency_measure`
arrives (or the panel acknowledges consumption). The GUI already applies the
value via `apply_latency_ms`; no panel logic changes beyond removing the
one-shot assumption.

### 3.5 Arduino: HID (Host-Shield passthrough) + WiFi

**Topology (the corrected model).** The Arduino plays two USB roles:

```
  Real mouse ─USB─▶ [USB Host Shield / MAX3421E] ─SPI(ICSP)─▶ Arduino R4 (RA4M1)
                                                                 │  merge
  Ragnarok PC ─command frames (HID report / serial / WiFi-UDP)─▶│  passthrough + aim
                                                                 │
  Arduino R4 native USB-C ─ONE combined HID mouse stream─▶ PC / game
```

- **Host role (shield):** `felis/USB_Host_Shield_2.0` (confirmed R4/RA4M1 support)
  reads the real mouse's HID reports over SPI/ICSP each poll.
- **Device role (native USB):** the RA4M1 presents a HID mouse to the PC; the game
  sees one mouse.
- **Merge:** each cycle the firmware emits `real_delta + injected_aim_delta` as a
  single mouse report. Real motion passes through 1:1; the aim delta rides the
  same physical device, bypassing Windows pointer ballistics entirely.

**Command channel (PC → Arduino aim deltas).** Same MAKCU frame
(`[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]`, `aim/protocol.py`) over any of:

1. **Raw-HID** vendor OUTPUT reports on the same native USB composite device —
   driverless (hidapi), no COM port. This is the "both directions over HID" path.
2. **USB-CDC serial** — existing `SerialTransport` (Leonardo/Pro-Micro users, or
   R4 CDC if configured as composite).
3. **WiFi/UDP** via the ESP32-S3 running a UDP-listener firmware that forwards
   frames to the RA4M1 over the internal UART. Honest caveat: ESP32 UDP bunches
   packets (~200 ms bursts), so WiFi is a **config/convenience** channel, not the
   low-latency aim path.

**PC-side changes (CI-testable behind fakes):**

- New `HidTransport` in `aim/arduino.py`: lazy `hidapi`/`hid` import (box-only),
  `write(frame_bytes)` sends a HID OUTPUT report carrying the frame. Interface is
  the same `write()`/`open()`/`close()` contract as `SerialTransport`/`UdpTransport`,
  so `ArduinoDriver` is unchanged. Frame chunking to the report size is handled in
  the transport.
- `ArduinoConfig.transport` gains `"hid"`; add `vid`/`hid_pid` (default 0).
  `build_arduino_transport` **requires** both to be set for the `hid` transport
  (raises `RuntimeError` on 0, like the serial/udp guards) — no usage-page
  auto-select, which would need enumerating all HID devices. `usage_page` is a
  fixed code constant (0xFF00).
- `build_arduino_transport` routes `"hid"` → `HidTransport` (guards required
  fields before the lazy import, like the others).
- Fix §2.4 wiring: `main()` owns **one** mouse driver, injected into both the
  fire component and the measure path; on config swap the reloader `close()`s the
  outgoing driver before opening the new one, with rollback if the new build
  fails; `app.aboutToQuit` closes the driver (and releases any held button).

**Firmware (box-only, rewritten for the R4 + Host Shield):**

- `firmware/ragnarok_mouse_r4/` — RA4M1 sketch:
  - USB Host Shield: read real-mouse HID reports (16-bit-per-axis descriptor so
    fast flicks aren't clipped to ±127).
  - Native USB: HID mouse OUT with a matching 16-bit report; emit
    `passthrough + injected` each cycle.
  - Command parser: accept MAKCU frames from a vendor HID OUTPUT report **and**
    from `Serial1` (the ESP32 link). Mirror `protocol.py` framing/CRC exactly.
  - Keep the DIAG echo for HIL latency.
- `firmware/ragnarok_esp32_udp/` — ESP32-S3 firmware: UDP server → forward frames
  to the RA4M1 over UART. Replaces the stock USB-serial bridge (documented flash
  steps; advanced/optional).
- Keep the existing `firmware/ragnarok_mouse/ragnarok_mouse.ino` (32u4/CDC) for
  Leonardo/Pro-Micro-over-serial users; note it does not do passthrough.

### 3.6 Correctness cleanup (fixes 2.5)

- Snapshot `det = self._det` once at the top of `tick()`; use it for both
  `detect()` and the `observe_lock` feature check (mirrors the existing
  `aim = self._aim` TOCTOU fix). Add a top-level `try/except` in `run()` that
  logs and surfaces a dead-loop state instead of silently exiting.
- Give bettercam `grab()` a bounded wait and make `WorkerThread.stop` also stop
  the capturer so a static screen can't hang shutdown. Minimal form: `stop()`
  stops the capturer (unblocking the waiter) before `join`; optionally a bounded
  `get_latest_frame` wait so aim/trigger keep ticking on static frames.

### 3.7 Trigger bot rework — independent, reliable (fixes 2.6)

Make the trigger a first-class, independently-toggled behaviour that fires on
crosshair-over-enemy whether or not auto-aim is engaged.

- **Decouple from the aim key.** Restructure `AimController.update` so the trigger
  section runs **every tick** (gated only by its own `trigger_active()` toggle),
  *before* the aim-active early-return; the aim-assist section (selection, IMM
  lead, aimer step, cursor move) stays gated by `aim.enabled && aim_active()`.
  `_disengage` resets aim state without releasing the trigger. One component owns
  the mouse, so there is no double-fire.
- **Unify the two paths.** `AimController` becomes the single fire/aim component
  whenever `aim.enabled || trigger.enabled`. When `aim.enabled` is false it still
  ticks the trigger; the standalone `TriggerController` is retired (or reduced to
  a thin alias) so there is one trigger implementation and one recoil path.
- **Fire on crosshair-containment.** The trigger targets the ENEMY track whose
  hitbox contains the crosshair (ROI centre), not the FOV-selected aim target —
  the correct test for "the crosshair is pointed at an enemy." With friend/foe
  off, `AllEnemyClassifier` already tags every detection ENEMY, so it "just
  works"; with friend/foe on, only classified enemies are shot (safety preserved).
- **Reliable delay.** `trigger.activation_delay_ms` default `80 → 35`; the
  eligibility timer tolerates brief occlusion (a configurable
  `max_occlusion_frames`, default ~2) instead of resetting on any single coasted
  frame, so flickery detections still fire. `require_line_clear` unchanged.
- **Recoil path** moves into the (now always-run) trigger section, so recoil
  compensation and full-auto pacing work with auto-aim off too.

## 4. Data flow (unchanged seams)

Capture → detect → track (IMM/GMC) → classify → `AimController.update` →
aimer.step (now commit + settle) → shaper → mouse driver (SendInput **or**
Arduino transport) → `CommandedMotionBuffer.push`. The only new data path is the
`HidTransport` alternative behind the existing `ArduinoDriver`; everything else is
an in-place behaviour change at an existing seam.

## 5. Error handling

- All new transports guard required config before the lazy import and raise
  `RuntimeError` with an actionable message (matching `build_arduino_transport`).
- The single-driver ownership + `close()`-on-swap removes the port-contention
  failure; the reloader rolls back to the previous controller if a rebuild
  raises, and `apply_config_change` continues to keep the GUI alive.
- Firmware CRC-rejects malformed frames (existing behaviour); UDP loss of a MOVE
  frame is self-healing (next frame supersedes); a lost BUTTON frame is the one
  case worth a firmware-side heartbeat/repeat — noted, not required for v1.

## 6. Testing

TDD, CI-safe (no GPU/Windows/serial/HID/MCU in tests):

- Aimer commit/settle math: table tests per aimer proving (a) no overshoot past
  remaining distance, (b) settle deadzone zeroes sub-`settle_px` motion,
  (c) `commit=1.0` reproduces prior flick/hybrid/predictive output, (d) feedback
  unchanged. Keep the §15 step-response regression green.
- Schema defaults: assert new default values (incl. `aim.toggle=True`, toggle-key
  defaults, `activation_delay_ms=35`) and that existing TOML round-trips.
- Trigger rework: trigger fires independently of aim-active state; fires on
  crosshair-contained ENEMY; retires/aliases `TriggerController` with the same
  observable behaviour; delay tolerates `max_occlusion_frames`; toggle-mode
  activation via a `FakeKeyProvider` rising edge; recoil applied on fire with aim
  off. Toggle state is exposed for telemetry.
- Timestamp plumbing: `Frame.t_capture_ns` sourced from an injected arrival clock;
  assert IMM `dt` uses arrival deltas (fake capturer).
- Latency latch: `tick()` keeps `latency_ms` across publishes until a new request;
  panel-consumption test.
- `HidTransport`: framing/chunking against a fake HID device (records reports);
  `build_arduino_transport` routing + guard-before-import for `"hid"`.
- Single-driver wiring: `main()` builds one driver; reloader closes the old before
  opening the new; rollback on build failure; close-on-exit releases buttons.
- TOCTOU + shutdown: `tick()` single-reads the detector; `WorkerThread.stop`
  stops the capturer.

Box-only (user verifies on hardware): real SendInput/HID/serial/UDP I/O; the R4
firmware (passthrough, native HID, vendor-report command channel); the ESP32 UDP
firmware; HIL latency; live overshoot/snappiness feel.

## 7. Non-goals / deferred

- Approach B (decoupled high-rate aim thread) — deferred; no "slow to reach"
  complaint.
- GMC fixed-window → per-frame-interval window (`egomotion.py`) — separate
  tracked follow-up; not required for the overshoot fix.
- Recoil-into-Smith-window refinement (recoil counts currently enter the commanded
  buffer) — minor; noted, not changed.
- VID/PID/vendor spoofing on the R4 native USB (forked USB stack) — out of scope.
- BLE / ESP-NOW transports — out of scope (UDP covers "socket/WiFi").

## 8. Rollout

Implemented as small TDD increments in dependency order: (1) aimer commit/settle
+ schema defaults, (2) trigger rework + toggle activation (aim + trigger) + state
telemetry, (3) latency latch + timestamp, (4) TOCTOU/shutdown cleanup, (5)
single-driver wiring + `HidTransport` + config, (6) firmware (box-only).
Each increment is independently testable and mergeable. Firmware lands last and is
flashed/verified by the user.
