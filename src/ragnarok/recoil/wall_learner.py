"""Recoil-on-wall learner (spec §11).

Turns a full-auto spray captured against a flat wall into a per-shot spray
pattern via global optical flow (phase correlation): as the gun kicks the view,
the wall texture translates in the frame; the negated cumulative translation is
the crosshair drift the compensator must counter.

Pure/testable here (``measure_shift`` uses cv2.phaseCorrelate, deterministic on
synthetic shifts; ``accumulate_drift`` + ``resample_at_shots`` are plain math).
The live frame capture during the spray is box-only — see scripts/learn_recoil.py.
"""
from __future__ import annotations

import numpy as np


def measure_shift(prev_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation (dx, dy) of the scene from ``prev`` to ``cur``."""
    import cv2
    a = prev_gray.astype(np.float32)
    b = cur_gray.astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    return (float(dx), float(dy))


def accumulate_drift(shifts) -> tuple[tuple[float, float], ...]:
    """Per-frame scene shifts -> cumulative VIEW-kick drift (px).

    The scene moves opposite the view, so each view kick is the negated scene
    shift; the drift is its running sum."""
    cx = cy = 0.0
    out: list[tuple[float, float]] = []
    for sx, sy in shifts:
        cx -= float(sx)
        cy -= float(sy)
        out.append((cx, cy))
    return tuple(out)


def resample_at_shots(cumulative, dt_frame_s: float, rps: float,
                      num_shots: int) -> tuple[tuple[float, float], ...]:
    """Sample the per-frame cumulative drift at shot times (i / rps) via linear
    interpolation -> a per-shot cumulative spray pattern for RecoilCompensator."""
    if not cumulative or rps <= 0.0 or num_shots <= 0 or dt_frame_s <= 0.0:
        return ()
    n = len(cumulative)
    xs = [i * dt_frame_s for i in range(n)]
    cxs = [c[0] for c in cumulative]
    cys = [c[1] for c in cumulative]
    total_t = (n - 1) * dt_frame_s
    pattern: list[tuple[float, float]] = []
    for i in range(num_shots):
        t = i / rps
        if t > total_t:
            break
        pattern.append((float(np.interp(t, xs, cxs)), float(np.interp(t, xs, cys))))
    return tuple(pattern)
