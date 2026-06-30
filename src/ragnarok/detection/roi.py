"""Dynamic-ROI (SEARCH/TRACK) coordinate math + FSM + planner (spec §5.2).

Pure: decides what region of the captured frame to feed the fixed 384 engine
(wide letterbox in SEARCH; tight square crop upscaled in TRACK) and maps the
engine-space detections back to full-frame coordinates. The actual pixel
crop/resize + the real detector are box-only (worker integration).
"""
from __future__ import annotations


def letterbox_params(src_w: int, src_h: int, dst: int) -> tuple[float, float, float]:
    scale = min(dst / src_w, dst / src_h)
    pad_x = (dst - src_w * scale) / 2.0
    pad_y = (dst - src_h * scale) / 2.0
    return (scale, pad_x, pad_y)


def map_back_letterbox(box, scale: float, pad_x: float, pad_y: float):
    x1, y1, x2, y2 = box
    return ((x1 - pad_x) / scale, (y1 - pad_y) / scale,
            (x2 - pad_x) / scale, (y2 - pad_y) / scale)
