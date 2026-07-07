"""Measure the aim feedback-loop latency at a flat wall (Windows, box-only).

Oscillates the view horizontally via SendInput and watches the wall's optical
flow respond; the lag at peak cross-correlation is the round-trip latency. Sets
aim.deadtime_ms (Smith predictor) and tracking.tau_render_s (feed-forward GMC).

Run:  .venv/Scripts/python.exe scripts/measure_latency.py [duration_s]
Then IMMEDIATELY click into your game and aim at a large, flat, textured wall.
The view will jitter left/right during the measurement — that's expected.
"""
from __future__ import annotations

import sys
import time

from ragnarok.config.store import load_config, save_config
from ragnarok.app import _config_path
from ragnarok.capture.factory import create_capturer
from ragnarok.aim.mouse import SendInputMouseDriver
from ragnarok.aim.latency_measure import WallLatencyMeasurer


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

    lag = WallLatencyMeasurer(cap, mouse, duration_s=dur).run()
    cap.stop()
    if lag is None:
        print("too few frames / low optical-flow signal — use a more textured wall")
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
