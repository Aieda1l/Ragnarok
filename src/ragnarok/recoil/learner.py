"""Recoil-on-wall pattern learner (spec §11).

Pure analysis: given the per-shot crosshair drift measured while holding fire at a
flat wall (from global optical flow — box-only capture), the cumulative sum is the
spray pattern the ``RecoilCompensator`` plays back. Averaging several sprays makes
the learned pattern robust to noise.
"""
from __future__ import annotations


def estimate_recoil_pattern(kicks) -> tuple[tuple[float, float], ...]:
    """Cumulative (dx, dy) crosshair drift per shot from per-shot kick vectors.

    ``kicks[i]`` is how far the view drifted between shot i-1 and shot i; the
    returned point i is the *cumulative* drift after shot i (what the compensator
    stores, then negates to counter)."""
    cx = cy = 0.0
    pts: list[tuple[float, float]] = []
    for dx, dy in kicks:
        cx += float(dx)
        cy += float(dy)
        pts.append((cx, cy))
    return tuple(pts)


def average_patterns(patterns) -> tuple[tuple[float, float], ...]:
    """Element-wise mean of several cumulative patterns (truncated to the
    shortest). Empty input -> ()."""
    patterns = [p for p in patterns if p]
    if not patterns:
        return ()
    n = min(len(p) for p in patterns)
    out: list[tuple[float, float]] = []
    for i in range(n):
        sx = sum(p[i][0] for p in patterns)
        sy = sum(p[i][1] for p in patterns)
        out.append((sx / len(patterns), sy / len(patterns)))
    return tuple(out)
