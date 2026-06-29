"""Pure single-class detection metrics (spec §12 step 5, §15).

IoU, VOC all-point Average Precision at a fixed IoU threshold (mAP@0.75 for the
single 'player' class == AP), and mean matched center-error. All pure numpy,
tested against analytically-known cases — no engine/GPU/frames.
"""
from __future__ import annotations

import math

import numpy as np


def _xyxy(box):
    # accept either a plain xyxy or a (xyxy, score) pair
    if len(box) == 2 and hasattr(box[0], "__len__"):
        return box[0]
    return box


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def average_precision_at_iou(preds, gts, *, iou_thresh: float = 0.75) -> float:
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    matched = [False] * len(gts)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for rank, i in enumerate(order):
        box = preds[i][0]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched[best_j] = True
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # VOC all-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_error(preds, gts, *, iou_thresh: float = 0.5) -> float | None:
    pred_boxes = [_xyxy(p) for p in preds]
    matched = [False] * len(pred_boxes)
    dists: list[float] = []
    for g in gts:
        best_iou, best_i = 0.0, -1
        for i, p in enumerate(pred_boxes):
            if matched[i]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_i >= 0 and best_iou >= iou_thresh:
            matched[best_i] = True
            gc, pc = _center(g), _center(pred_boxes[best_i])
            dists.append(math.hypot(gc[0] - pc[0], gc[1] - pc[1]))
    if not dists:
        return None
    return float(sum(dists) / len(dists))
