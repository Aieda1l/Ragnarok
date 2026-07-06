"""Learn a recoil spray pattern by firing at a flat wall (Windows, box-only).

Aim at a large, flat, textured wall, run this, and HOLD FIRE for the capture
window. It measures the view kick per frame (phase correlation over the captured
ROI) and prints a per-shot spray pattern to paste into the GUI Recoil tab.

Run:  .venv/Scripts/python.exe scripts/learn_recoil.py [duration_s] [rps]
"""
from __future__ import annotations

import sys
import time

import cv2

from ragnarok.config.store import load_config
from ragnarok.app import _config_path
from ragnarok.capture.factory import create_capturer
from ragnarok.recoil.wall_learner import measure_shift, accumulate_drift, resample_at_shots
from ragnarok.gui.recoil_model import format_pattern_text


def main() -> None:
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    cfg = load_config(_config_path())
    rps = float(sys.argv[2]) if len(sys.argv) > 2 else (cfg.recoil.fire_rate_rps or 10.0)

    cap = create_capturer(cfg.capture)
    cap.start()
    print(f"Aim at a flat wall. HOLD FIRE for {dur:.1f}s starting in 3...")
    for i in (3, 2, 1):
        print(i, "...")
        time.sleep(1)
    print("FIRE!")

    frames = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        f = cap.grab()
        if f is not None:
            frames.append((time.perf_counter(), cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)))
    cap.stop()
    print(f"captured {len(frames)} frames")
    if len(frames) < 3:
        print("too few frames — check capture / target_fps")
        return

    shifts = [measure_shift(frames[i - 1][1], frames[i][1]) for i in range(1, len(frames))]
    drift = accumulate_drift(shifts)
    dt = (frames[-1][0] - frames[0][0]) / (len(frames) - 1)
    pattern = resample_at_shots(drift, dt, rps, int(dur * rps))

    print(f"\n--- learned {len(pattern)}-shot pattern (rps={rps:g}) ---")
    print(format_pattern_text(pattern))
    print(f"\nPaste into the Recoil tab's spray-pattern box; set Fire rate = {rps:g}.")
    print("If the compensation pulls the WRONG way, negate the scale or flip the signs.")


if __name__ == "__main__":
    main()
