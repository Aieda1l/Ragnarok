"""Head-aware aim-point resolution (spec §6.1).

When the detector has a dedicated head class (e.g. a model trained with
``enemy_head``), ``aim_point == "detected_head"`` aims at the actual detected head
box rather than the ``head_frac`` heuristic on the body box. Pure + testable:
given the target track and all tracks it finds the head belonging to the target
(a head-class track whose centre lies inside the target's box, smallest wins), or
falls back to the head-fraction point when no head detection is available.
"""
from __future__ import annotations

from ragnarok.core.types import Track
from ragnarok.aim.fov import aim_point


def _best_head_for(target: Track, tracks, head_class_id: int) -> Track | None:
    if target.class_id == head_class_id:      # the target itself is a head detection
        return target
    x1, y1, x2, y2 = target.xyxy
    best: Track | None = None
    best_area: float | None = None
    for tr in tracks:
        if tr.class_id != head_class_id:
            continue
        cx, cy = tr.center
        if x1 <= cx <= x2 and y1 <= cy <= y2:   # head centre inside the target body box
            hx1, hy1, hx2, hy2 = tr.xyxy
            area = (hx2 - hx1) * (hy2 - hy1)
            if best is None or area < best_area:  # smallest = most head-like
                best, best_area = tr, area
    return best


def resolve_aim_point(target: Track, tracks, *, mode: str, head_frac: float,
                      head_class_id: int) -> tuple[float, float]:
    """The aim point for ``target`` (screen/ROI px).

    ``"detected_head"`` aims at the associated detected head box when one exists,
    else falls back to the ``head_frac`` point; ``"head"``/``"body"`` behave as
    ``aim.fov.aim_point``.
    """
    if mode == "detected_head":
        head = _best_head_for(target, tracks, head_class_id)
        if head is not None:
            return head.center
        return aim_point(target, head_frac, "head")
    return aim_point(target, head_frac, mode)
