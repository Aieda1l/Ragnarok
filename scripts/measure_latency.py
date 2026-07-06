"""Measure the aim feedback-loop latency at a flat wall (Windows, box-only).

Oscillates the view horizontally via SendInput and watches the wall's optical
flow respond; the lag at peak cross-correlation is the round-trip latency. Sets
aim.deadtime_ms (Smith predictor) and tracking.tau_render_s (feed-forward GMC).

Run:  .venv/Scripts/python.exe scripts/measure_latency.py [duration_s]
Then IMMEDIATELY click into your game and aim at a large, flat, textured wall.
The view will jitter left/right during the measurement — that's expected.
"""
from __future__ import annotations

import math
import sys
import time

import cv2

from ragnarok.config.store import load_config, save_config
from ragnarok.app import _config_path
from ragnarok.capture.factory import create_capturer
from ragnarok.aim.mouse import SendInputMouseDriver
from ragnarok.aim.latency import estimate_lag
from ragnarok.recoil.wall_learner import measure_shift


def main() -> None:
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
    cfg = load_config(_config_path())
    cap = create_capturer(cfg.capture)
    cap.start()
    mouse = SendInputMouseDriver(compensate_ballistics=cfg.input.compensate_ballistics)
    mouse.connect()

    print("Click into your game and aim at a flat wall. Measuring in 3...")
    for i in (3, 2, 1):
        print(i, "...")
        time.sleep(1)
    print("measuring (view will jitter left/right)...")

    amp, freq = 40.0, 3.0          # counts amplitude, Hz — a clean oscillation to correlate
    prev_gray = None
    prev_pos = 0.0
    commanded, observed, times = [], [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        f = cap.grab()
        if f is None:
            continue
        t = time.perf_counter() - t0
        gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            dx, _ = measure_shift(prev_gray, gray)
            observed.append(dx)
            times.append(t)
            pos = amp * math.sin(2.0 * math.pi * freq * t)
            commanded.append(pos - prev_pos)          # per-frame velocity of the sinusoid
            prev_pos = pos
            mouse.move_relative(commanded[-1], 0.0)
        prev_gray = gray
    cap.stop()

    n = min(len(commanded), len(observed))
    if n < 10:
        print("too few frames — check capture / target_fps")
        return
    dt = (times[-1] - times[0]) / (len(times) - 1)
    lag = estimate_lag(commanded[:n], observed[:n], dt, max_lag_frames=int(0.25 / dt))
    if lag is None:
        print("could not estimate (low optical-flow signal) — use a more textured wall")
        return

    ms = round(lag * 1000.0, 1)
    print(f"\nmeasured round-trip latency: {ms} ms  (frame dt {dt * 1000:.1f} ms)")
    new = cfg.model_copy(update={
        "aim": cfg.aim.model_copy(update={"deadtime_ms": float(ms)}),
        "tracking": cfg.tracking.model_copy(update={"tau_render_s": float(lag)}),
    })
    save_config(new, _config_path())
    print(f"saved aim.deadtime_ms={ms}  tracking.tau_render_s={lag:.4f}")
    print("Restart ragnarok (or it hot-reloads on the next config edit).")


if __name__ == "__main__":
    main()
