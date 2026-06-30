"""Dynamic-ROI (SEARCH/TRACK) coordinate math + FSM + planner (spec §5.2).

Pure: decides what region of the captured frame to feed the fixed 384 engine
(wide letterbox in SEARCH; tight square crop upscaled in TRACK) and maps the
engine-space detections back to full-frame coordinates. The actual pixel
crop/resize + the real detector are box-only (worker integration).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


def letterbox_params(src_w: int, src_h: int, dst: int) -> tuple[float, float, float]:
    scale = min(dst / src_w, dst / src_h)
    pad_x = (dst - src_w * scale) / 2.0
    pad_y = (dst - src_h * scale) / 2.0
    return (scale, pad_x, pad_y)


def map_back_letterbox(box, scale: float, pad_x: float, pad_y: float):
    x1, y1, x2, y2 = box
    return ((x1 - pad_x) / scale, (y1 - pad_y) / scale,
            (x2 - pad_x) / scale, (y2 - pad_y) / scale)


def crop_region_for(center, size: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    cx, cy = center
    x0 = int(round(cx - size / 2.0))
    y0 = int(round(cy - size / 2.0))
    x0 = max(0, min(x0, frame_w - size))
    y0 = max(0, min(y0, frame_h - size))
    return (x0, y0, size, size)


def map_back_crop(box, crop_region, dst: int):
    x0, y0, size, _ = crop_region
    r = size / dst
    x1, y1, x2, y2 = box
    return (x1 * r + x0, y1 * r + y0, x2 * r + x0, y2 * r + y0)


class RoiMode(str, Enum):
    SEARCH = "search"
    TRACK = "track"


class RoiState:
    def __init__(self, *, max_missed: int, rescan_interval: int) -> None:
        self._max_missed = max_missed
        self._rescan = rescan_interval
        self._mode = RoiMode.SEARCH
        self._missed = 0

    @property
    def mode(self) -> RoiMode:
        return self._mode

    def update(self, *, has_lock: bool) -> RoiMode:
        if self._mode == RoiMode.SEARCH:
            if has_lock:
                self._mode = RoiMode.TRACK
                self._missed = 0
        else:  # TRACK
            if has_lock:
                self._missed = 0
            else:
                self._missed += 1
                if self._missed >= self._max_missed:
                    self._mode = RoiMode.SEARCH
                    self._missed = 0
        return self._mode

    def wants_rescan(self, frame_index: int) -> bool:
        return (self._mode == RoiMode.TRACK and self._rescan > 0
                and frame_index % self._rescan == 0)


@dataclass(frozen=True)
class RoiPlan:
    mode: RoiMode
    region: tuple[int, int, int, int]     # (x0, y0, w, h) in full-frame pixels
    letterboxed: bool                     # True = SEARCH (letterbox), False = TRACK (crop)


class DynamicRoiPlanner:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._dst = cfg.model_input_px
        self._state = RoiState(max_missed=cfg.max_missed_frames,
                               rescan_interval=cfg.rescan_interval_frames)

    def plan(self, *, frame_w: int, frame_h: int, target_center,
             frame_index: int, has_lock: bool) -> RoiPlan:
        self._state.update(has_lock=has_lock)
        if self._state.mode == RoiMode.SEARCH or self._state.wants_rescan(frame_index):
            return RoiPlan(mode=self._state.mode, region=(0, 0, frame_w, frame_h),
                           letterboxed=True)
        if target_center is None:
            raise ValueError("target_center is required for a TRACK crop plan")
        region = crop_region_for(target_center, self._cfg.track_roi_size, frame_w, frame_h)
        return RoiPlan(mode=self._state.mode, region=region, letterboxed=False)

    def map_back(self, box, plan: RoiPlan):
        if plan.letterboxed:
            x0, y0, w, h = plan.region
            scale, pad_x, pad_y = letterbox_params(w, h, self._dst)
            mx1, my1, mx2, my2 = map_back_letterbox(box, scale, pad_x, pad_y)
            return (mx1 + x0, my1 + y0, mx2 + x0, my2 + y0)
        return map_back_crop(box, plan.region, self._dst)
