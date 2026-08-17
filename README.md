# Ragnarok

A vision-based aim system for a **self-authored, single-player, offline shooter sandbox**.
Everything it knows about the game comes from the screen: a TensorRT RF-DETR detector finds
players in a centered ROI, a motion-only BoT-SORT tracker gives them stable identities, an HSV
outline gate separates enemies from teammates, and a control loop turns the selected target's
pixel error into mouse counts.

It is a CV + control-systems + embedded + PySide6 project as much as it is an aimbot. The
interesting parts are the Smith predictor that kills dead-time rubber-banding, the feed-forward
ego-motion compensator, the relay/Nelder-Mead PID auto-tuner, and the USB-Host-Shield
passthrough firmware.

**Scope:** a closed environment — an offline game the author wrote, no other players, no online
service. Everything here assumes that context.

---

## Status

| | |
|---|---|
| Tests | **754 passing** (`pytest`, no GPU / no hardware required) |
| Detector | RF-DETR-Small, locally trained — val **mAP@50 ≈ 0.85**, **mAP@50:95 ≈ 0.56** |
| Throughput | **~152 FPS** detect on an RTX 3090 via TensorRT (~50 FPS torch FP16, ~0.2 FPS on CPU) |
| Platform | Windows 11 + NVIDIA (capture, SendInput and raw-input paths are Win32) |

---

## Pipeline

```
 screen ──▶ capture ──▶ detect ──▶ track ──▶ friend/foe ──▶ select ──▶ aim ──▶ mouse
            bettercam   RF-DETR    BoT-SORT   HSV ring       sticky     PID +    SendInput
            (DXGI)      TRT/torch  + GMC      + vote         FOV cone   Smith    or Arduino
                                                                        predictor
                             │                                    │         │
                             └────── dynamic ROI feedback ────────┘         │
                                                                            ▼
                                          telemetry ──▶ GUI dashboard + lock-on overlay
```

One worker thread owns the loop (`worker/loop.py`); the Qt GUI never blocks it. Config edits
swap an immutable `AppConfig` snapshot and rebuild only the affected components in place, so
every knob is live-tunable while the loop runs.

---

## Features

**Detection**
- RF-DETR-Small (Apache-2.0 variants only), two classes: `enemy` (body) and `enemy_head`.
- Two backends: `rfdetr_torch` (auto FP16 fuse) and `rfdetr_trt` (shape-driven TensorRT
  runtime — reads engine I/O shapes, so a retrained model with a different class count needs
  no code change). Validated against torch to <0.1 px.
- Live confidence slider and hot-swap detector reload (engine swap without restarting).
- Optional **dynamic ROI**: a SEARCH/TRACK FSM crops tight around the locked target and
  upscales it into the same 384 engine, so distant 6-px targets stay detectable. Every forward
  transform has a tested exact inverse.

**Tracking**
- Vendored lean motion-only BoT-SORT / ByteTrack core (from MIT `NirAharon/BoT-SORT`; ReID and
  CMC stripped, scipy assignment, Mahalanobis 2-DOF gate χ²=5.9915). numpy + scipy only.
- **Feed-forward GMC**: injected mouse counts are back-projected into a 2×3 affine and applied
  to the tracker's ego-motion, τ_render-aligned — so your own turning doesn't look like target
  motion. Off by default; needs `deg_per_count` + `tau_render_s` from calibration.
- Per-track IMM (CV + CA, filterpy) for lead prediction.

**Friend / foe**
- Single detection class plus a runtime HSV outline-ring gate and a temporal vote (≥3 of 5).
  Changing the game's enemy-outline color needs no retrain.
- Built-in palettes (`default`, `wong` colorblind) plus an **eyedropper** — click the live
  preview to sample any in-game color into a custom HSV band.
- Disabled → every detection is treated as an enemy.

**Aim**
- Four aimers: **Flick** (constant-speed, follows the live target), **Feedback** (P / PI / PID
  with quadratic creep zone and three-fold anti-windup), **Hybrid** (feedback then a final
  flick, for low-RoF weapons), **Predictive** (velocity feed-forward).
- **Smith predictor** (`aim.deadtime_ms`) — the root fix for rubber-banding. The loop advances
  its notion of the crosshair by counts already commanded but not yet visible, so it stops
  re-issuing corrections for moves in flight.
- Sticky target selection: inner acquire FOV / outer retain FOV, dwell time, switch margin, and
  a rule that a head-class box inside an enemy body box never competes as its own candidate
  (that oscillation was a rubber-band source).
- Aim point: geometric `head` (fraction of box height), `body`, or `detected_head` — the actual
  detected head box belonging to the target, falling back to the fraction when no head is seen.
- WindMouse humanized motion shaping, adaptive lead, sub-pixel accumulation, `commit` fraction
  and `settle_px` deadzone, optional true rectilinear (pinhole) px→deg.

**Trigger & recoil**
- Trigger bot fires when the crosshair is inside an enemy hitbox, with an activation delay, a
  line-clear gate, and occlusion tolerance (N coasted frames still fire).
- Recoil compensation: GUI-editable per-shot spray patterns, full-auto per-shot advance at a
  configured fire rate, and an **optical-flow wall learner** that derives the pattern by
  spraying at a flat wall.

**Activation model**
- Aim and trigger are **independent ON/OFF toggles** on non-obtrusive keys
  (`VK_XBUTTON2` / `VK_XBUTTON1` by default). No hold-to-aim; the trigger fires on
  crosshair-in-enemy whether or not aim assist is on.

**Interface**
- PySide6, Cyberpunk 2077 design language — frameless window, custom title bar, red/cyan
  palette, and custom widgets (`SegmentedToggle`, `CyberSlider`, `ArrowSelector`, `ChromeFrame`).
- Seven tabs: **Dashboard** (FPS / p50 / p99 sparklines), **Aim**, **Targeting**
  (detection + tracking + friend/foe + eyedropper), **Fire** (trigger + recoil),
  **Calibrate**, **Interface** (keybinds, overlay, input, profiles), **Advanced**
  (diagnostics, motion).
- Click-through smart-weapon **lock-on overlay** (Win32 `WS_EX_TRANSPARENT|LAYERED|TOPMOST`):
  FOV brackets, per-target diamonds, lock highlight, tracking line, confidence readout.
- Named config profiles plus portable import/export.

**Diagnostics**
- Step-response measurement (rise / overshoot / settling / dead time) against a simulated plant
  or live, relay auto-tuning (Ziegler–Nichols seed) and Nelder-Mead ITAE numeric tuning.
- One-click **latency measurement**: oscillates the view at a flat wall, cross-correlates
  commanded motion against optical flow, and writes both `aim.deadtime_ms` and
  `tracking.tau_render_s`.
- Hardware-in-the-loop round-trip latency over the Arduino DIAG echo.

---

## Requirements

- Windows 11, NVIDIA GPU (developed on an RTX 3090 / Ampere)
- Python ≥ 3.11
- The game running **borderless-windowed** on the captured monitor

Torch, `rfdetr` and TensorRT are **not** in `pyproject.toml` dependencies — they are per-machine
CUDA builds. Install them yourself.

## Install

```powershell
uv sync                              # numpy, pydantic, PySide6, bettercam, opencv, scipy, filterpy…

# CUDA torch — match your driver (this box: driver 595.71 / CUDA 13.2 -> cu130)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 `
  --index-strategy unsafe-best-match

uv pip install rfdetr                # detector
uv pip install tensorrt-cu13 onnx onnxscript   # optional: TensorRT backend (~152 FPS)
```

Verify CUDA actually landed — a CPU-only torch wheel is the difference between 150 FPS and 0.2:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Optional, only for the Arduino paths: `pip install pyserial` (serial transport) or
`pip install hidapi` (driverless raw-HID transport).

> **OneDrive note:** this checkout lives under OneDrive, whose sync locks files during large
> installs (`Access is denied` on torch). Exclude `.venv/`, `dataset/` and `output/` from sync,
> or clear the stale directory and retry.

## Run

```powershell
uv run ragnarok
```

Config is created on first launch at `%APPDATA%\Ragnarok\config.toml`; profiles live beside it
in `%APPDATA%\Ragnarok\profiles\`.

---

## Calibrate first

Uncalibrated, the loop gain is a guess and the crosshair will oscillate. Two measurements fix it,
both from the **Calibrate** tab:

1. **Sensitivity** — press Reset, do exactly one full 360° turn in game, press Set. This reads
   raw HID counts (`WM_INPUT`, unaffected by the Windows pointer-speed slider) and sets both
   `aim.sensitivity` and `tracking.deg_per_count` to 360/counts.
2. **Latency** — press Measure latency, click into the game and aim at a large flat textured
   wall. The view jitters for ~2.5 s, then `aim.deadtime_ms` and `tracking.tau_render_s` are
   written for you.

Then set `aim.screen_width_px` to your real monitor width (the app warns on mismatch) and, if you
want ego-motion compensation, flip `tracking.gmc` to `feedforward` — it is inert without step 1
and 2, and warns on startup if so.

If it still overshoots, raise `aim.deadtime_ms`; if it undershoots, lower it.

**Input driver:** for games that read raw input (most FPS), leave `input.compensate_ballistics`
off — they bypass Windows pointer ballistics, so compensating over-moves. For the finest
precision, put the Windows pointer slider on the middle notch (speed 10, the 1:1 multiplier),
disable "Enhance pointer precision", or use the Arduino path.

---

## Configuration

TOML, validated by frozen pydantic models (`config/schema.py`). Every section is exposed in the
GUI and hot-reloads.

| Section | Covers |
|---|---|
| `capture` | backend (`bettercam`/`mss`), ROI size, target FPS, monitor index |
| `detection` | backend (`rfdetr_torch`/`rfdetr_trt`), model size, confidence, FP16, engine path |
| `tracking` | BoT-SORT thresholds, GMC mode, `deg_per_count`, `tau_render_s` |
| `classification` | palette, enemy color, custom HSV band, ring thickness, vote window |
| `aim` | aimer, gains, FOV cone, dwell, aim point, sensitivity, dead time, lead |
| `motion` | WindMouse shaping parameters |
| `recoil` | enabled, scale, per-shot pattern, fire rate |
| `trigger` | key, activation delay, occlusion tolerance, line-clear, button |
| `diagnostics` | step-response and relay-tuning parameters |
| `training` | frame/dataset/engine dirs, sampling thresholds, Roboflow project |
| `arduino` | transport (`serial`/`udp`/`hid`), port, host, VID/PID |
| `input` | mouse driver (`sendinput`/`arduino`), ballistics compensation |
| `overlay` | what the lock-on overlay draws, plus cosmetic scanline/chroma FX |
| `calibration` | Calibrate-tab hotkeys |
| `dynamic_roi` | SEARCH/TRACK crop sizes, miss counter, rescan interval |

Secrets are never config fields — the Roboflow key is read from
`RAGNAROK_ROBOFLOW_API_KEY`.

---

## Training pipeline

Two YOLO datasets are merged into one COCO dataset, RF-DETR is fine-tuned locally, and the
checkpoint is exported to a TensorRT engine the app is then pointed at.

```powershell
uv run python scripts/prepare_dataset.py   # YOLO x2 -> COCO {enemy, enemy_head}, 80/10/10 split
uv run python scripts/train.py             # RF-DETR-Small, 100 epochs -> output/
uv run python scripts/export_engine.py     # best EMA ckpt -> ONNX -> .engine, updates config
uv run python scripts/export_engine.py --int8   # optional PTQ (Q/DQ ONNX, TRT 11 INT8)
```

`dataset/`, `output/` and `engines/` are gitignored (large and box-specific).

A Roboflow path also exists (`training/roboflow_client.py` — upload, download versions, and an
active-learning hard-example miner) for anyone who prefers hosted labeling; set
`RAGNAROK_ROBOFLOW_API_KEY` plus `training.roboflow_workspace`/`project`.

---

## Arduino output (optional)

SendInput is the default and needs no hardware. The Arduino path exists so mouse motion arrives
as real HID reports rather than injected events.

The real topology is a **passthrough**, not a standalone HID device: your physical mouse plugs
into a USB Host Shield on an UNO R4, the R4 merges that motion with Ragnarok's aim deltas, and
presents **one** combined HID mouse to the PC over its native USB-C.

```
  Real mouse ─USB─▶ [USB Host Shield / MAX3421E] ─SPI─▶ Arduino R4 (RA4M1)
                                                          │  merge
  Ragnarok PC ─command frames (HID / serial / UDP)──────▶ │  passthrough + aim
                                                          │
  R4 native USB-C ─ONE combined HID mouse stream─▶ PC / game
```

Wire protocol (`aim/protocol.py`, mirrored by the firmware and verified in CI):
`[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]`, CRC-8 poly 0x07 / init 0x00 over CMD+LEN+PAYLOAD.
Commands: MOVE, BUTTON, CONFIG, DIAG (echoes `micros()` for HIL latency).

Three sketches ship in `firmware/` — R4 passthrough, an ESP32-S3 UDP→UART bridge, and a legacy
standalone 32u4 HID mouse. See [`firmware/README.md`](firmware/README.md) for flashing, the
command-channel options, and the one remaining box-only piece (a vendor-collection descriptor
edit in the Renesas core needed for the driverless raw-HID transport).

---

## Scripts

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Merge the YOLO datasets into one COCO dataset |
| `train.py` | Fine-tune RF-DETR-Small locally |
| `export_engine.py` | Checkpoint → ONNX → TensorRT engine (`--int8` for PTQ) |
| `mouse_counts.py` | Live raw HID mouse-count monitor (360° sensitivity method) |
| `measure_latency.py` | Wall optical-flow round-trip latency → dead time + τ_render |
| `learn_recoil.py` | Spray at a wall → per-shot recoil pattern |
| `measure_hil.py` | PC↔MCU round-trip latency over the Arduino DIAG echo |

---

## Development

```powershell
uv run pytest                 # 754 tests
```

The whole suite is CI-safe: no GPU, no game, no MCU, no network. That is deliberate — every
Windows/CUDA/hardware boundary sits behind an injected seam (fake capturers, fake detectors,
fake transports, fake clocks), so the control math, wire codec, ROI transforms, tracker
association and Qt models are all testable off the box. Widgets are exercised offscreen with
`pytest-qt`. Only genuinely physical behavior — real DXGI capture, SendInput, hardware timing —
is verified by hand.

### Layout

```
src/ragnarok/
  capture/         bettercam (DXGI) + mss capturers, injectable clock
  detection/       RF-DETR torch + TensorRT backends, ONNX/engine export, dynamic ROI
  tracking/        vendored BoT-SORT core, feed-forward GMC, calibration solvers
  classification/  HSV ring friend/foe, temporal vote, eyedropper
  aim/             aimers, PID, IMM, selection, FOV, mouse drivers, Arduino protocol
  trigger/         trigger bot
  recoil/          compensator, pattern learners
  motion/          WindMouse shaping
  latency/         stage profiler, adaptive lead
  diagnostics/     step response, relay + numeric auto-tuning
  training/        benchmarking, frame sampling, Roboflow client
  gui/             PySide6 panels, overlay, theme, custom widgets
  worker/          the loop
  config/          pydantic schema, TOML store, profiles
firmware/          three .ino sketches (R4 passthrough, ESP32 bridge, 32u4 legacy)
scripts/           training and calibration tools
docs/superpowers/  design spec + per-phase implementation plans
tests/             754 tests
```

### Design docs

- Spec: [`docs/superpowers/specs/2026-06-26-ragnarok-design.md`](docs/superpowers/specs/2026-06-26-ragnarok-design.md)
- Phase plans: [`docs/superpowers/plans/`](docs/superpowers/plans/)

---

## Third-party

The tracker core is vendored from **MIT-licensed `NirAharon/BoT-SORT`** (not AGPL BoxMOT), with
ReID, CMC and `lap` removed. RF-DETR is used in its Apache-2.0 variants only. Aimer taxonomy
(flick / feedback / hybrid / trigger) follows `AccessViolationEnjoyer/NeuralBot`.
