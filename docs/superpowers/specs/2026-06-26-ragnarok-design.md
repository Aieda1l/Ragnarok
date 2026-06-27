# Ragnarok — Design Specification

**Date:** 2026-06-26
**Status:** Draft for review — revision 4 (adds error-gated anti-windup, recoil-into-ego-motion, HIL latency diagnostics, experimental 2:4 sparsity)
**Type:** Real-time computer-vision + control-systems application

## Scope & context

Ragnarok is a **vision-based aim system for a single-player, offline shooter sandbox game that the user wrote themselves**. There are no other players, no online play, and no anti-cheat in the loop. It is a computer-vision, control-theory, embedded, and GUI engineering project that targets the user's own game. Every design decision assumes this closed, self-owned environment. "Humanization," hardware HID, and transport options are treated as control-quality / input-method engineering choices, not detection-evasion features.

This document is grounded in four rounds of technical research (detection, tracking, capture, input, control, friend/foe CV, GUI; transport latency/jitter; and feasibility verification of the proposed enhancements). Source pointers are in the References section.

---

## 1. Goals and non-goals

### Goals
- Detect game characters on screen in real time with **RF-DETR**, including a **custom training pipeline** built around **Roboflow**.
- Distinguish **enemies from teammates** by the game's configurable colored outline (purple/red/yellow, colorblind-aware) **without retraining**.
- Track multiple targets stably through **occlusion and fast camera panning** (BoT-SORT / ByteTrack / DeepSORT, selectable).
- Provide multiple, switchable **aiming methods**: Flick, Feedback (P/PI/PID), Hybrid, Predictive, plus Trigger Bot, FOV gating, recoil compensation, and humanized motion.
- Offer **smart target selection** that is sticky enough to not jitter but trivially easy to switch targets on demand.
- Support **software mouse injection** and **Arduino UNO R4 WiFi + USB Host Shield** hardware HID, over **USB-CDC, UDP/WiFi, BLE, and ESP-NOW** transports.
- Ship a **modern PySide6 GUI** styled after **Cyberpunk 2077**, with a **smart-weapon-lock-on-style FOV overlay**, full live tuning, telemetry, diagnostics, and calibration wizards.
- Be **measurable**: per-stage latency budgets, step-response system identification, PID auto-tune, and FP16/INT8 accuracy benchmarking.

### Non-goals
- No online/multiplayer use; no anti-cheat evasion engineering.
- No reliance on game memory/internals — the game **cannot expose ground-truth telemetry**, so the runtime is fully vision-based and the dataset comes from Roboflow.
- No support for exclusive-fullscreen overlay (a universal OS constraint; the game runs **borderless-windowed**).

---

## 2. Operating context & constraints

| Constraint | Value / decision |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB, Ampere). TensorRT FP16 default; INT8 optional (no FP8 on Ampere). |
| Topology | **Single PC** (game + capture + inference + aim + Arduino-over-USB). WiFi/ESP-NOW designed in but optional. |
| Display | Refresh rate **auto-detected** (`QScreen.refreshRate()` / Windows API); adaptive loop. Game **must run borderless-windowed** for the overlay. |
| Capture | DXGI Desktop Duplication via **BetterCam/DXcam** (NVFBC is dead: GeForce-restricted + Win10/11-deprecated). |
| Ground truth | **None** — dataset labeled in Roboflow; runtime is black-box CV. |
| OS / language | Windows 11, Python 3.11+ (`perf_counter_ns` = QPC), PySide6. |

---

## 3. High-level architecture

**One process, two threads** (cross-process frame serialization would destroy latency):

- **Worker thread (hot loop):** owns all CUDA/TensorRT/capture state. Runs `capture → detect → track → classify → select → aim → shape → output`, plus recoil and trigger, at the panel's refresh cadence.
- **GUI thread:** read-only observer + config editor. Receives a **published-latest immutable telemetry snapshot** (polled by a 60 Hz `QTimer`, no signals on the hot path); pushes **immutable config snapshots** down (atomic reference swap, no locks on the hot read path).
- **Overlay:** separate frameless, click-through, always-on-top window driven by the same telemetry snapshot.

```
                       ┌──────────────────────── GUI thread (Cyberpunk 2077 UI) ───────────────────────┐
                       │ Control panels · pyqtgraph telemetry · profiles · wizards · diagnostics        │
                       │ Overlay (smart-lock FOV reticle, target brackets, tracks, confidence)          │
                       └──── poll latest snapshot (QTimer 60Hz) ◄──── │ ──── config snapshot swap ▼ ─────┘
┌──────────────── Worker thread (hot loop @ panel refresh, all CUDA here) ────────────────────────────────┐
│ Capture(ROI, t_capture stamp) → Detector(RF-DETR/TRT, search|track engine) → Tracker(BoT-SORT+GMC)      │
│   → Friend/Foe(HSV outline gate + temporal vote) → Target Select(FOV + score + hysteresis)              │
│   → IMM filter (lead) → Aimer(Flick/PID/Hybrid/Predictive) → Motion Shaper(WindMouse) → MouseDriver     │
│                              └→ Recoil compensator     └→ Trigger bot (safety-gated)                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The **hot loop never blocks on the GUI**. The overlay is **cosmetic/diagnostic only** — aim uses detection coordinates, never rendered pixels — so overlay composition latency cannot affect accuracy.

---

## 4. Module breakdown

Package layout (`src/ragnarok/`). Each module has one responsibility and a narrow interface so it can be tested in isolation.

```
ragnarok/
  app/            # entry point, DI wiring, worker/GUI lifecycle
  core/           # Frame, Detection, Track, Target dataclasses; event/telemetry bus; clock
  config/         # schema (pydantic), TOML load/save, profiles, hot-reload, snapshot swap
  capture/        # Capturer interface; BetterCamCapturer (ROI) default, MssCapturer fallback, optional ZeroCopyCapturer (C++/pybind DXGI->CUDA)
  detection/      # Detector interface; RFDETRTensorRT (search/track engines), RFDETROnnx, RFDETRTorch
  tracking/       # vendored lean BoT-SORT/ByteTrack core (owns motion+association); Mahalanobis gate; optional pinned-BoxMOT backend
  egomotion/      # camera yaw/pitch integration (world-space frame); feed-forward "active GMC" affine (tau_render-aligned) from commanded+passthrough deltas
  classification/ # FoeClassifier (HSV outline ring + temporal vote); color calibration
  targeting/      # FovGate, TargetSelector (acquire by angle, retain by Mahalanobis), aim-point logic
  filtering/      # MotionFilter interface; CV/CA Kalman; IMMFilter (filterpy or Numba lean); lead/extrapolation
  aim/            # Aimer interface; FlickAimer, FeedbackAimer(P/PI/PID), HybridAimer, PredictiveAimer
  motion/         # MotionShaper interface; WindMouseShaper, BezierPerlinShaper, NullShaper
  output/         # MouseDriver interface; SendInput, Interception, Arduino{Serial,Udp,Ble,EspNow}
  recoil/         # RecoilCompensator; per-weapon pattern tables; wall-learner
  trigger/        # TriggerBot (safety gates)
  latency/        # timestamping, per-stage profiler, adaptive lead estimator
  diagnostics/    # step-response runner, relay/numeric PID auto-tuner
  training/       # frame grabber, Roboflow client, train/export(ONNX/TRT/INT8), benchmark
  gui/            # PySide6 app: theme, panels, widgets, overlay, telemetry plots
  firmware/       # Arduino sketches (configurable matrix) + PC-side protocol lib (shared)
  telemetry/      # snapshot types, ring/pool buffers, logging
```

---

## 5. Vision pipeline

### 5.1 Capture
- **BetterCam/DXcam** (DXGI Desktop Duplication), **centered ROI** sized to the active detector input (`region=`, ~1 MiB at 512²), `copy=False`/`grab_view` to skip the extra CPU memcpy, background-threaded "pull latest," then `torch.from_numpy(frame).cuda(non_blocking=True)`. For a small ROI the H2D transfer is **sub-0.5 ms — not the bottleneck**.
- Each frame is stamped at grab with `t_capture = time.perf_counter_ns()` (QPC), carried through the whole pipeline for adaptive lead and per-stage profiling.
- **Optional `ZeroCopyCapturer` (profiling-gated, off by default):** a small **C++/pybind11** DXGI→CUDA-interop backend (`cudaGraphicsD3D11RegisterResource` → map → `cudaMemcpy2DFromArray` into a reused linear buffer → DLPack→torch). Saves only **~0.2–0.5 ms** for a small ROI and adds keyed-mutex sync + a **LUID constraint** (capture adapter must equal the torch CUDA device — asserted at startup). GPU preprocessing is shared across both backends. No mature pure-Python path exists (CuPy lacks D3D11 interop).
- Fallback: `python-mss` if DXGI init fails.

### 5.2 Detection (RF-DETR)
- **RF-DETR-Small** default (Nano for max FPS), single **`player`** class. ONNX → **TensorRT FP16** engine via RF-DETR's exporter.
- **Dynamic-ROI (single-engine default):** one **static 384×384** engine (one `IExecutionContext`), **CUDA-Graph-captured** for the fixed shape.
  - State machine: **`SEARCH`** letterbox/downscale the wide ROI → 384; on lock, **`TRACK`** crop a tight ROI around the predicted target and **bilinear-upscale it to 384** — same engine, but the target fills the frame → maximum pixels-on-target and better localization (what aim precision needs) at no inference-speed cost. Revert to `SEARCH` on track loss / N missed frames; **periodic wide rescan** while tracking so incoming enemies aren't missed. The tracker predicts the crop center one frame ahead.
  - **Optional "max-FPS" mode:** two resident fixed-shape engines (384 search + 256 track), each its own context, switched by enqueuing on one shared CUDA stream. Switching is cheap and VRAM-trivial on 24 GB, but it trades per-target detail for only a small (~sub-ms–1 ms on the 3090) speedup and complicates CUDA-Graph capture. Off by default.
- **GPU preprocessing (no `cv2.cuda`):** standard pip OpenCV wheels are CPU-only, so crop/letterbox/resize/normalize run **on-GPU in torch** (`pin_memory().to('cuda', non_blocking=True)` → tensor slice → `F.pad` → `F.interpolate` → normalize), handed to TensorRT via `data_ptr()` with no host round-trip. **One shared CUDA stream + one preallocated device pool** (torch caching allocator); never `cudaMallocManaged` in the loop (page-fault latency spikes).
- **INT8 (optional, opt-in, profiling-gated):** RF-DETR is **transformer-dominated** (DINOv2 ViT + shallow Deformable-DETR decoder; the only conv is the patch-embed stem), so there is no "CNN backbone" to quantize and the realistic INT8 speedup on Ampere is **small (~1.05–1.3×, sometimes nil after Q/DQ overhead)** — **FP16 is the production default.** If pursued: **explicit Q/DQ via NVIDIA TensorRT Model Optimizer (`modelopt`)** — the legacy implicit-quant calibrators (`IInt8EntropyCalibrator2`/`MinMax`/`Legacy`, `setDynamicRange`) are **deprecated in TRT 10.4 and removed in TRT 11**, so don't build on them. **Selective boundary:** Q/DQ on the **linear/MLP GEMMs** (Q/K/V, FFN, output projections) + optionally the patch-embed conv; **keep FP16** on attention scores + softmax, LayerNorm, GELU, and the bbox-regression head (these saturate under uniform INT8). Validate engine layer precisions (Polygraphy), end-to-end latency, and `mAP@0.75`/center-error vs FP16; accept only if the speedup clears a threshold and mAP drop stays in budget, else revert. **QAT** (modelopt) is the recovery path if PTQ loses too much. Given the marginal gain, spend optimization budget on pipeline / CUDA-graphs / resolution first.
- **2:4 structured sparsity (experimental only):** available via `modelopt` `mts.sparsify` + `trtexec --sparsity=enable` (FP16, Ampere Sparse Tensor Cores), but for this transformer-dominated net the realistic **end-to-end gain is only ~1.1–1.15× at batch 1** (the 1.5–1.8× figure is per-GEMM; attention/softmax/LN/GELU/decoder aren't accelerated) and it **requires sparsity-aware fine-tuning** to hold mAP (one-shot magnitude pruning collapses transformer accuracy). Kept as a benchmark-harness experiment, **not** a default; sparse-INT8 (~1.26–1.47× in the ViT literature) only with distillation-based QAT. FP16-dense remains the baseline.
- Portable fallbacks: ONNX Runtime and Torch.

### 5.3 Tracking
- **Vendored lean tracker:** a forked ~300-line **BoT-SORT/ByteTrack core** lives in-repo so we own the motion + association stage (current upstream BoxMOT master is a C++ native backend that cannot accept a per-frame affine or a custom gate from Python). Keeps `STrack` + matching utils; the per-track filter is our **IMM** (§6.2). Uniform `Tracker.update(dets, frame, ego_affine) -> tracks` interface.
- **Feed-forward "active GMC" (default):** instead of CV optical-flow GMC (ORB/ECC + RANSAC, several ms/frame in Python), **back-project the known camera motion** — the sum of injected aim deltas **and** the player's physical mouse deltas (read via the USB Host Shield passthrough) — into a 2×3 affine applied to each track's predicted state before association (the standard `multi_gmc` transform: rotation/scale on pos+vel, translation on pos; `R = I` for pure yaw/pitch). O(1), no feature extraction. Back-projection: `f_px = (W/2)/tan(HFOV/2)`, `yaw° = Δcounts · deg_per_count`, `t_x = −f_px·tan(yaw°)` (same `f_px` for `t_y`); upgrade to a 3×3 homography only if wide-FOV edge drift appears. Sign and `deg_per_count` are calibrated empirically (turn across a static target; the post-GMC prediction residual should collapse). **CV-GMC stays optional** to mop up residual non-mouse camera motion (recoil view-kick, knockback, animations).
- **Temporal phase alignment (τ_render):** the camera responds to a mouse delta only after the game's render+display latency (≈1–3 frames). So the affine for frame *N* is built by integrating the **timestamped mouse-delta ring buffer** over the **render-time window** `[t_capture − τ_render − Δframe, t_capture − τ_render]`, not the most-recent deltas — critical during fast flicks. `τ_render` is a **calibratable slider**, **auto-estimated** by cross-correlating commanded motion against measured global optical flow.
- **Mahalanobis association gate (2-DOF, χ²=5.9915):** on top of IoU, reject pairs with `d² = (z−Hx)ᵀ S⁻¹ (z−Hx) > 5.9915`, `S = HPHᵀ+R` from the track's own KF — velocity/uncertainty-aware, and reliable once feed-forward GMC makes predicted centers accurate during flicks. (4-DOF χ²=9.4877 available if aspect/height predictions are also trusted.)
- **Behavior:** default association = IoU + Mahalanobis gate, **Re-ID OFF** (skins match; costs latency/VRAM). Selectable: **ByteTrack** "fast" mode, an optional Re-ID experiment, and a **pinned pure-Python BoxMOT (v10.0.83)** backend for comparison/benchmarking (CV-GMC via a monkeypatched `cmc` object). Raised `track_buffer` (coast through smoke/walls); lowered low-confidence threshold (recover occluded boxes).

### 5.4 Friend/foe classification (teammate safety core)
- **Hybrid model:** one detector `player` class + a **runtime HSV outline-ring gate** + **temporal vote (≥3 frames)** before a track is labeled `enemy`. This satisfies "configurable/colorblind outline color, no retrain."
- Ring sampling: `dilate(mask) − erode(mask)` band around the silhouette; require a **fraction** of pixels matching the active enemy color (not a single pixel); morphological open to drop particle speckle.
- Red uses **two hue ranges** (OpenCV hue wraparound). High S/V floors reject desaturated UI/background.
- Colors stored as named HSV band tuples; **Wong colorblind-safe defaults**; **one-click eyedropper calibration** regenerates bands from the live screen.

---

## 6. Targeting & aim system

### 6.1 FOV gating & target selection
- **Pixel-space FOV cone**: `fov_radius_px` derived from desired degrees and the game's horizontal FOV; engage only candidates within radius.
- **Acquisition** (which enemy to engage) = `argmin` of a weighted cost: **angle-to-crosshair (dominant)**, then **occlusion/visibility**, then **distance**, with **head-vs-body aim-point** logic (top-of-box head bias when confident) — this encodes *your* intent.
- **Retention** (staying locked) uses the **Mahalanobis distance** from the locked track's innovation covariance `S` rather than raw pixels, so the lock rides the target's (stretched-covariance) trajectory and rejects Euclidean-near clutter / adjacent targets.
- **Hysteresis for stable-but-switchable locks:** **dual-radius sticky FOV** (acquire radius < retain radius), a **dwell timer (~150 ms)**, and a **switch margin (~20%)** before stealing to a new target. The lock resets instantly on re-trigger / target death / FOV exit / occlusion timeout — so manual switching stays immediate.

### 6.2 Motion filtering & prediction (IMM)
- **Per-target IMM** (`filterpy.kalman.IMMEstimator`), **2-model default**: CV (low process noise, smooth steady tracking) + CA/high-q (snappy on jukes). `mu=[0.5,0.5]`, `M=[[0.95,0.05],[0.05,0.95]]`. Optional 3rd coordinated-turn model (config).
- Shared state layout across member filters; **F and Q rebuilt every frame from real inter-frame dt**; **R data-driven** from measured RF-DETR box-center jitter (scaled by box size).
- **Occlusion / reacquire:** predict-only during gaps; on reacquire or track switch, **re-init from measurement, zero velocity/accel, inflate P, reset mode probabilities** (avoids stale-velocity overshoot).
- Selectable simpler models (single CV / CA) for comparison.
- **Coordinate frame:** the aim target's filter runs in **world-angular space** (camera yaw/pitch integrated from mouse deltas) so its velocity is ego-motion-free by construction — see §6.4. The feed-forward GMC affine (§5.3) supplies the pixel-space predict warp the tracker's association needs.
- **Performance path:** start with filterpy; if the filter stage shows in the p99 profile, swap in a **Numba `@njit` lean IMM** — a flat, pre-allocated, contiguous-array state machine (`states (M,N)`, `covs (M,N,N)`, `mu (M,)`, transition matrix) with one jitted `mix → predict → update → prob-update` function and jitted F/Q builders (or a **dt-bucket LUT**, 144/240 Hz + lerp). Target **<30 µs** per target-update, single-threaded. The `MotionFilter` interface keeps this swappable.

### 6.3 Aimers (two decoupled layers)

**Layer A — control law (what delta to command):**
- **Flick Aimer** — acquire position once, smooth move; immune to dropped detections; best on stationary targets.
- **Feedback Aimer** — a **2-DOF feedforward + feedback** controller: `u = Kp·e + Ki·∫e + Kd·ė + Kff·v̂`, where `v̂` is the IMM velocity estimate. The `Kff·v̂` term commands the target's velocity directly so the reactive PID only cleans up acceleration/jukes/model error — killing constant-velocity tracking lag without a dangerous `Ki`. Selectable P / PI / PID; **error filtered before the D term**. **Anti-windup is three-fold:** *conditional integration* (integrate only when `|e|` is below a threshold / crosshair inside the hitbox), an *integral-contribution clamp*, and *freeze-on-actuator-saturation* (stop integrating once commanded `u` hits the mouse driver's max step). Because the `Kff·v̂` term already carries velocity tracking, `Ki`'s role shrinks — the default leans **PD + FF with a small I**, so windup is rarely even triggered. (Predictive lead pushes the *setpoint* ahead, `Kff` matches *velocity* to it — both use `v̂`; the GUI flags not to double-count when tuning them together.) `v̂` is **low-pass / alpha-beta smoothed and velocity-clamped** before the `Kff` term to prevent feed-forward runaway (§6.4).
- **Hybrid Aimer** — proportional approach, final flick when close; ideal for snipers / low rate-of-fire.
- **Predictive Aimer** — `aim = imm_pos + imm_vel × t_lead` (+ accel term from the CA model); iterative time-of-flight solve for projectile weapons.

**Layer B — motion shaping (how the cursor travels):**
- **WindMouse** default (gravity/wind/`M_0`/`D_0` exposed; low wind for combat precision); optional **Bézier + low-amplitude Perlin** tremor; `NullShaper` for raw deltas.
- Large deltas split into per-poll sub-deltas (chunk >127 px for the int8-limited Arduino HID path).

### 6.4 Ego-motion isolation & control-loop stability
The feed-forward velocity term (`Kff·v̂`) carries a stability risk: any residual ego-motion the active-GMC subtraction misses is read by the filter as *target* velocity, which `Kff` then amplifies → positive feedback (jitter/runaway). Three guards:
- **World-space (inertial) filtering — the root fix:** the aim target's IMM state is maintained in **world-angular coordinates** (camera yaw/pitch integrated from injected + passthrough mouse deltas), so its velocity is **ego-motion-free by construction** and the loop is broken at the source rather than patched after measurement. Each detection's screen position is converted to a world angle (via `f_px`, FOV) before the filter update; the predicted world angle is converted back to a screen position / required mouse delta to aim. Data **association stays in pixel space** (IoU needs pixel boxes), fed by the active-GMC predict warp.
- **Feed-forward smoothing:** `v̂` passes through a low-pass / alpha-beta filter before the `Kff` term to damp high-frequency gain spikes.
- **Velocity saturation:** the `Kff`-commanded velocity is clamped to the plausible maximum target speed, so a bad estimate can't drive a runaway.
- **Recoil folded into ego-motion:** the learned per-shot recoil kick `(Δθ, Δφ)` is injected into the **same camera-orientation integrator** (τ_render-aligned at `t_fire`), so the tracker attributes the on-screen view-kick to *recoil*, not target motion — the world-space target estimate stays static through a spray (see §6.6).

### 6.5 Adaptive lead estimation
- `t_lead = (now − t_capture)` [true per-frame age] **+ EWMA(actuation + transport latency)**, recomputed each frame — not a fixed constant. Self-corrects under CPU/GPU/USB scheduling jitter and feeds the predictive aimer + IMM.

### 6.6 Recoil compensation
- **Per-weapon learned `(dx, dy, dt)` offset tables**, advanced one entry per shot, reset on fire-release.
- **Fed into the camera state, not just the output:** beyond emitting the counter-move, the known kick is added to the camera-orientation integrator (§6.4), so recoil is part of the ego-motion model and never pollutes world-space target-velocity estimation during a spray.
- **Wall-learner wizard:** fire **N = 5–10 full magazine dumps** at a flat wall, track impact/crosshair drift (bright-pixel centroid / template match). **Align each run by the software fire timestamp `t_fire`** (instant and jitter-free, unlike muzzle-flash detection) and **resample the displacement onto a uniform ~1 kHz continuous timeline** relative to `t_fire` using **PCHIP / monotone (or linear) interpolation — not natural cubic, which overshoots the sharp muzzle-rise onset**. Then aggregate the **median** Δx/Δy across runs to isolate the deterministic recoil curve from stochastic bloom, and store the **MAD/spread** as a determinism/quality metric. Scale by the sensitivity→pixels calibration.
- Muzzle-flash CV used **only** as an optional "is-firing" signal, never as the primary compensation source.

### 6.7 Trigger bot
- Fires only when **all** hold: enemy-confirmed (color + temporal vote) **AND** crosshair inside the hitbox **AND** configurable activation delay (50–150 ms) **AND** burst-with-recheck.
- **Never fires on a coasted/predicted (occluded) box.** Between bursts, re-checks pixels along the crosshair-to-target line; a teammate-colored pixel or occluder **aborts** the burst. Usable standalone (no aim assist).

---

## 7. Safety & robustness

- **Occlusion:** coast the track for **aim continuity**, decay confidence per missed frame, but **disable the trigger** whenever the box is predicted/coasted — never shoot a ghost behind cover.
- **False-positive suppression (4 layers):** confidence threshold (~0.4–0.6) · `min_hits` temporal confirmation · size/aspect-ratio sanity gate · **static ignore-masks** for the player's own weapon/hands + HUD regions.
- **Particle/lighting robustness:** ring-fraction (not single-pixel) color test + morphological cleanup + track-level temporal vote so a smoke/flash can't flip a teammate to enemy.
- **Global panic/disable hotkey**; aim only while an **aim key** is held (or toggle); safe defaults on first run.

---

## 8. Output: mouse drivers, transports, firmware

### 8.1 PC-side `MouseDriver` interface
`move_relative(dx, dy)`, `set_button(btn, down)`, `connect()/close()`. Delta-chunking and pacing live in a shared base class. Backends:

| Backend | Notes |
|---|---|
| **SendInputDriver** | **Primary default.** In-process, sub-ms, no polling cap. Relative deltas (no ABSOLUTE flag). |
| **InterceptionDriver** | Fallback for games that read Raw Input and ignore SendInput (signed driver + reboot). |
| **ArduinoSerialDriver** | USB-CDC binary frame. (Baud is irrelevant on the R4's native USB; the 1 ms USB frame is the floor.) |
| **ArduinoUdpDriver** | "Socket" / WiFi transport (per user's tutorial firmware). |
| **ArduinoBleDriver** | BLE, XOR-obfuscated `WriteWithoutResponse`; **config-rate only** (~50–150 Hz ceiling). |
| **ArduinoEspNowDriver** | ESP-NOW via a PC-side ESP32 dongle; best wireless (~1–4 ms, low jitter). |

> **Honest note:** the UNO R4 WiFi native HID commonly caps near **125 Hz**, and adding WiFi UDP + USB Host Shield to the same 48 MHz loop drops passthrough to ~600–900 Hz. The clean 1000 Hz path requires `bInterval=1` **plus the ArduinoCore-renesas USB.cpp delay-removal patch (PR #331)** with no host shield in the loop. Software injection has no such cap — hence it is the default; the Arduino is the "real-HID-device + physical-mouse-passthrough" option.

### 8.2 Wire protocol (all transports)
Single compact binary frame, MAKCU-modeled, shared by PC lib and all MCU firmware:

```
[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]
CMD 0x01 MOVE   : dx:i16, dy:i16, buttons:u8, mode:u8     (mode = DIRECT | DEVICE_INTERP)
CMD 0x02 BUTTON : mask:u8
CMD 0x03 CONFIG : transport/mode/interp params
CMD 0x04 DIAG   : HIL latency echo — MCU returns hardware-timer Δµs (packet-receipt → HID-report assertion)
```

### 8.3 Configurable firmware (single codebase, compile-time axes via `#ifdef`)
- **TRANSPORT:** `USB_CDC | UDP | BLE | ESP_NOW`
- **ENDPOINT:** `RA4M1_HID` (default, lowest latency) | `LEONARDO_I2C` (Wire @400 kHz, ~0.3–0.6 ms) | `ESP32_DIRECT` (TinyUSB, bypasses internal UART bridge)
- **MOVEMENT MODE:** `DIRECT` | `DEVICE_INTERP` — device-side target accumulation + EWMA smoothing + per-step jitter at the 1 kHz HID tick (from the user's BLE sketch). This decouples PC update rate from output smoothness and is the **default for WiFi/BLE/ESP-NOW** transports. Interpolation runs on the MCU that owns the HID endpoint.
- **PASSTHROUGH:** USB Host Shield (MAX3421E, SPI) reads a physical mouse and merges its deltas; on/off.

### 8.4 QoL / advanced firmware features
- **Polling-rate mod:** `bInterval=1` **(corrected from the tutorial's 0, which is invalid for a FS interrupt endpoint)** + the USB.cpp delay-removal patch; verify with a polling-rate tester.
- **COM-port hiding:** `-DDISABLE_USB_SERIAL` in `boards.txt` `build.defines` + matching guard in `SerialUSB.cpp` (fragile — must match the installed core's guard names; removes CDC upload → reflash via DFU/double-tap).
- **HID descriptor / VID:PID spoofing (e.g. Razer Viper Mini, VID 0x1532):** requires a **forked, version-pinned ArduinoCore-renesas + bootloader** editing TinyUSB `usb_descriptors.c` (device descriptor + HID report descriptor). `boards.txt` alone only changes the IDE label. Provided as an optional build variant.

---

## 9. Latency, timestamping & telemetry

- **Timestamps:** `time.perf_counter_ns()` (QPC), stamped at capture and propagated inside the frame payload; integer ns math; per-stage age computed everywhere. Powers adaptive lead and the latency HUD.
- **Per-stage profiler:** capture / preprocess / inference / postprocess / track / classify / select / aim / send, reported as **p50/p99**.
- **Worker→GUI telemetry:** worker builds an **immutable snapshot** (frozen dataclass) and publishes via a single reference assignment; GUI polls it on a **60 Hz QTimer** (race-safe under the GIL; lossy/coalescing, which is correct for "latest value"). **No queued signals on the hot path** — those are reserved for low-rate discrete events.
- **Preview frame:** 2–3 buffer pool, published by reference; never paint from a buffer the worker may overwrite.

---

## 10. GUI — Cyberpunk 2077 design language

The detailed component design will be produced with the **UI/UX Pro Max skill** during implementation (with an optional live visual companion for mockups). This section fixes the direction and requirements.

### 10.1 Visual language
- **Aesthetic:** Night City / Cyberpunk 2077 — dark near-black backgrounds, **signature electric yellow (#FCEE0A)** primary accent, **cyan/teal and alert-red** secondaries, angular/asymmetric panels, corner-bracket framing, thin scanlines, subtle **glitch / chromatic-aberration** transitions, hexagon/diamond motifs, animated data readouts.
- **Typography:** condensed techy faces (free equivalents such as **Rajdhani / Saira / Chakra Petch**); monospaced numerals for telemetry.
- **Implementation in Qt:** QSS for base theming + **custom `QPainter` / `QOpenGLWidget` widgets** for the bracket frames, animated gauges, glitch effects, and the lock-on reticle; `pyqtgraph` for live plots. Fonts bundled and loaded at startup.

### 10.2 Overlay — "smart-weapon lock-on"
Styled after the CP2077 smart-link/target-lock reticle:
- **FOV indicator** as an arc/ring with tick marks.
- **Acquisition brackets** that converge/snap onto a detected enemy (the lock-on animation); **diamond markers** over confirmed targets; a distinct highlight for the **currently locked** target with a thin tracking line to the crosshair.
- **Enemy/teammate color-coding**, per-track confidence, off-screen target direction hints.
- Frameless, always-on-top, `Qt::Tool`, `WA_TranslucentBackground`, **`QOpenGLWidget`-backed** (avoids the slow GDI layered path), `WS_EX_TRANSPARENT | WS_EX_LAYERED` for click-through, rendered on its own timer decoupled from the hot loop. (D3D11/DComp backend is an optional cosmetic-only future add; it offers no latency/accuracy benefit.)

### 10.3 Information architecture (tabs/panels)
Dashboard (live preview + p50/p99 latency & FPS graphs) · Aim (aimer select, PID/WindMouse, FOV) · Targeting (scoring weights, hysteresis) · Detection (model/engine, thresholds, dynamic-ROI) · Tracking (tracker select, GMC, buffers) · Friend/Foe (color picker, eyedropper, colorblind presets) · Recoil (pattern editor + wall-learner) · Trigger · Input (driver/transport select, Arduino port/IP, test) · Diagnostics (step-response, auto-tune) · Training (Roboflow loop, export, benchmark) · Profiles · Wizards.

---

## 11. Diagnostics & calibration

- **Step-response / system ID:** command a known step; record response; compute **rise time (10–90%), overshoot %, settling time (±2%), dead time**. Two modes: **(a)** desktop OS-cursor (`GetCursorPos`) to characterize driver + shaper + transport in isolation; **(b)** in-game **closed-loop** using the detector as the position sensor (run against a **stationary dummy** for LTI validity); **(c) HIL** — the MCU echoes hardware-timer Δµs (packet-receipt → HID-report assertion) so device transport+firmware latency is isolated from the PC pipeline (`perf_counter_ns`) and the game-render portion (detector closed-loop).
- **PID auto-tune:** **relay feedback (Åström–Hägglund)** → `Ku = 4d/(πa)`, `Tu` from the limit cycle → Ziegler–Nichols seed (`Kp=0.6Ku`, `Ki=1.2Ku/Tu`, `Kd=0.075Ku·Tu`), biased toward PI/low-overshoot; plus a **numeric-optimization** fallback (Nelder-Mead/CMA-ES minimizing ITAE + overshoot + effort) over logged step data. Results are seeds for manual fine-tuning, not final.
- **Wizards:** sensitivity→pixels/deg calibration, outline-color eyedropper, recoil-on-wall learner, FOV calibration, **render-latency (τ_render) calibration** (auto-estimated by cross-correlating commanded motion with global optical flow; manual slider override), first-run setup.

---

## 12. Training pipeline (Roboflow)

1. **Frame grabber** in-app records ROI frames during play (smart sampling on detection uncertainty / scene change).
2. Upload to **Roboflow**; annotate the single `player` class (Label Assist + active learning).
3. Export **COCO**; `train` RF-DETR (defaults: 100 epochs, EMA, early-stopping; reduce batch + raise grad-accum if VRAM-bound).
4. Export **ONNX → build TensorRT engines** (FP16 default; optional INT8 with the calibration step); hot-swap into the detector.
5. **Benchmark harness:** FP16 vs INT8 (`mAP@0.75`, center-error) and RF-DETR vs YOLO11 vs tracker variants, on the user's own frames.
6. **Hard-example miner** pushes low-confidence/missed frames back to Roboflow for the next dataset version.

---

## 13. Configuration & profiles

- **Schema-validated** (pydantic) config; **TOML** files under `%APPDATA%/Ragnarok`.
- **Per-weapon / per-game profiles** (recoil table, FOV, smoothing, aimer, trigger), hot-reload via `QFileSystemWatcher` and in-GUI sliders — both funnel through the **immutable snapshot swap** to the worker.
- Optional **weapon auto-detect** (HUD CV) to auto-load the matching profile.
- Import/export, presets, and config versioning.

---

## 14. Error handling

- Capture init failure → fall back DXGI → mss; surfaced in GUI.
- Detector/engine load failure → fall back TRT → ONNX → Torch; clear status.
- Arduino disconnect / CRC error / transport timeout → auto-reconnect, driver health indicator, and **fail-safe to aim-disabled** (never a runaway).
- Config validation errors never reach the hot loop (validated on load; bad snapshot rejected).
- All faults logged (structured) and shown non-blockingly in the GUI.

---

## 15. Testing strategy

- **Unit:** FOV math, scoring/hysteresis, IMM (synthetic maneuvering tracks vs known ground truth), HSV gate (synthetic outline images incl. red wraparound + colorblind palettes), protocol encode/decode + CRC, delta-chunking, recoil table advance.
- **Component:** detector on a fixed frame set (mAP), tracker ID-switch/fragmentation across scripted pan/occlusion clips, driver against a loopback/echo MCU.
- **Integration:** recorded-gameplay replay through the full pipeline (deterministic, no game needed); latency profiler assertions.
- **System ID / control:** step-response regression (rise/overshoot/settling within bounds) on the desktop-cursor path.
- **Manual validation:** in-game checklist under particle effects, smoke, muzzle flash, and each colorblind palette before trusting the trigger gate.

---

## 16. Packaging / deployment

- **PyInstaller 6.x, `--onedir`** (CUDA/torch DLLs make `--onefile` slow/fragile); exclude unused Qt modules and the second Qt binding; add torch/CUDA hidden-imports.
- Consider shipping **TensorRT/ONNX Runtime** instead of full torch to shrink the runtime.
- Arduino firmware shipped as source + build instructions (forked core variant for spoofing documented separately).

---

## 17. Phasing / milestones

1. **Skeleton + capture + detection** — RF-DETR (FP16) on captured frames, GUI shell, timestamping + latency profiler, telemetry snapshot plumbing.
2. **Tracking + friend/foe** — BoxMOT/BoT-SORT+GMC, HSV gate + temporal vote, overlay with detections.
3. **Aim core** — FOV gate, target selection + hysteresis, IMM filter, Flick + P-controller, SendInput driver → first working aim.
4. **Full aim system** — all aimers + adaptive predictive lead + WindMouse + recoil + trigger with safety gates.
5. **Diagnostics** — step-response + relay/numeric auto-tune; dynamic-ROI two-engine path.
6. **Training pipeline** — Roboflow loop, ONNX/TRT/INT8 export, benchmark harness, hard-example miner.
7. **Arduino backends** — RA4M1 HID firmware (USB-CDC) → UDP/WiFi → ESP-NOW → BLE; passthrough; spoofing variant; PC-side drivers.
8. **Polish** — Cyberpunk UI pass (UI/UX Pro Max), smart-lock overlay, wizards, profiles, packaging.

---

## 18. Key risks & open questions

- **Raw Input:** does the game read raw input (ignoring SendInput)? Empirically test in Phase 3; Interception is the fallback. (Since the user wrote the game, a controllable input path is also an option.)
- **INT8 accuracy:** must be validated; likely mixed-precision. FP16 remains default.
- **Dynamic-ROI situational awareness:** mitigated by periodic wide rescan; tune the rescan cadence.
- **R4 WiFi 125 Hz / 600–900 Hz ceilings:** accept for hobby use, or pursue the USB.cpp patch / a different MCU if hardware output rate becomes the bottleneck.
- **Spoofing maintenance:** requires a version-pinned core fork; guard names drift between releases.
- **Tracker packaging:** upstream BoxMOT master is now a C++ native build that can't take a custom per-frame affine or gate from Python → we vendor a lean tracker; pin any optional BoxMOT dependency to a pure-Python release (v10.0.83), never `-U`.
- **Feed-forward GMC correctness:** depends on the `deg_per_count`/FOV calibration, the affine sign, and the **`τ_render` phase alignment** — validate empirically (turn across a static target; the prediction residual should collapse) before trusting it in the loop.
- **Feed-forward control stability:** the `Kff·v̂` loop can run away on residual ego-motion → mitigated by **world-space filtering + `v̂` smoothing + velocity clamp (§6.4)**; validate with the step-response harness.
- **INT8 is marginal here:** transformer-dominated RF-DETR yields only ~1.05–1.3× on Ampere, and the legacy TRT calibrators are removed in TRT 11 → **FP16 is the default**; INT8 only via explicit Q/DQ (`modelopt`) if FP16-latency-bound, and may net nothing after Q/DQ overhead. **2:4 sparsity is even more marginal** (~1.1–1.15× at batch 1, needs sparsity-aware fine-tuning) → experimental-only.
- **Zero-copy capture:** needs a custom C++/pybind extension and a same-LUID adapter for only ~0.2–0.5 ms on a small ROI → optional, not default.

---

## 19. Tech stack summary

Python 3.11+ · PySide6 · pyqtgraph · RF-DETR (`rfdetr`) · TensorRT (FP16 default + CUDA Graphs; optional INT8 explicit-Q/DQ + experimental 2:4 sparsity via NVIDIA Model Optimizer `modelopt`) / ONNX Runtime / PyTorch (on-GPU preprocessing) · `supervision` · vendored lean BoT-SORT/ByteTrack tracker (+ optional pinned pure-Python BoxMOT v10.0.83) · OpenCV (CPU) · NumPy · Numba · filterpy · BetterCam/DXcam (+ optional C++/pybind DXGI→CUDA zero-copy) · pydantic · tomlkit · WindMouse · Interception · Arduino (ArduinoCore-renesas, TinyUSB, USB_Host_Shield_2.0, WiFiS3, ArduinoBLE, ESP-NOW) · Roboflow.

---

## 20. References (selected, verified)

- RF-DETR: github.com/roboflow/rf-detr · rfdetr.roboflow.com (benchmarks, training, export/INT8)
- Trackers: ByteTrack (FoundationVision), BoT-SORT (NirAharon), BoxMOT (mikel-brostrom), Ultralytics tracking docs
- Capture: DXcam (ra1nty), BetterCam (RootKit-Org); NVFBC Win10 deprecation (NVIDIA TB-09382-001)
- Input/transport: MS SendInput/MOUSEINPUT docs, Interception (oblitum), Arduino UNO R4 WiFi docs, MAKCU API, kmbox NET, ElectricUI latency benchmark, ESP-NOW (Espressif), ArduinoCore-renesas PR #331, UNO-R4-WiFi-freedom
- Control/CV: WindMouse (ben.land), filterpy IMMEstimator, PID/relay auto-tuning (Åström–Hägglund), HSV inRange (OpenCV), Wong colorblind palette, CS2 recoil pattern
- GUI: Qt threading (pythonguis), Qt translucent/transparent overlay flags, DXGI flip-model + DirectComposition (MS Learn), pyqtgraph perf, PyInstaller + PySide6/torch
- Python timing: `perf_counter_ns` = QPC (PEP 418, CPython docs)
