"""Dynamic-ROI detector wrapper (spec §5.2).

Wraps any base ``Detector`` with a SEARCH/TRACK ROI: SEARCH feeds the whole
frame letterboxed to the model input; TRACK feeds a tight square crop around the
last locked target, upscaled — so a distant/small target fills more of the model
input and is detected. Detections are mapped back to full-frame (ROI) pixel
coords, so everything downstream (tracker/aim/overlay) is unchanged.

Opt-in via ``dynamic_roi.enabled`` (default off). The planner + coordinate math
(detection.roi) are pure/tested; only the crop/resize here touches pixels.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from ragnarok.core.types import Frame, Detections
from ragnarok.detection.base import Detector
from ragnarok.detection.roi import DynamicRoiPlanner, RoiPlan, letterbox_params


def render_roi(image: np.ndarray, plan: RoiPlan, dst: int) -> np.ndarray:
    """Crop ``image`` to ``plan.region`` and produce a ``dst``×``dst`` model input:
    letterbox (SEARCH, preserves aspect) or stretch the square crop (TRACK)."""
    x0, y0, w, h = plan.region
    sub = image[y0:y0 + h, x0:x0 + w]
    if plan.letterboxed:
        scale, pad_x, pad_y = letterbox_params(w, h, dst)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(sub, (rw, rh))
        out = np.zeros((dst, dst, sub.shape[2]), dtype=sub.dtype)
        px, py = int(round(pad_x)), int(round(pad_y))
        out[py:py + rh, px:px + rw] = resized
        return out
    return cv2.resize(sub, (dst, dst))


class DynamicRoiDetector(Detector):
    def __init__(self, base: Detector, planner: DynamicRoiPlanner, *, model_input_px: int) -> None:
        self._base = base
        self._planner = planner
        self._dst = model_input_px
        self._frame_index = 0
        self._center: tuple[float, float] | None = None
        self._has_lock = False

    def observe_lock(self, center, has_lock: bool) -> None:
        """Feed back the current locked-target centre (full-frame px) so the NEXT
        frame's plan can crop around it. Called by the worker after aim."""
        self._center = center
        self._has_lock = bool(has_lock)

    def detect(self, frame: Frame) -> Detections:
        h, w = frame.image.shape[:2]
        has_lock = self._has_lock and self._center is not None
        plan = self._planner.plan(frame_w=w, frame_h=h, target_center=self._center,
                                  frame_index=self._frame_index, has_lock=has_lock)
        self._frame_index += 1
        model_img = render_roi(frame.image, plan, self._dst)
        sub = Frame(image=model_img, t_capture_ns=frame.t_capture_ns, region=frame.region)
        raw = self._base.detect(sub)
        items = tuple(replace(d, xyxy=self._planner.map_back(d.xyxy, plan)) for d in raw)
        return Detections(items=items)
