"""Pure-numpy IoU, replacing the original cython_bbox dependency.

Derived from NirAharon/BoT-SORT (tracker/matching.py ious/iou_distance),
MIT licensed, reimplemented in plain numpy. numpy only.
"""
import numpy as np


def iou_batch(atlbrs, btlbrs):
    """Pairwise IoU between two sets of (x1, y1, x2, y2) boxes -> (A, B)."""
    a = np.ascontiguousarray(atlbrs, dtype=float).reshape(-1, 4)
    b = np.ascontiguousarray(btlbrs, dtype=float).reshape(-1, 4)
    ious = np.zeros((a.shape[0], b.shape[0]), dtype=float)
    if ious.size == 0:
        return ious

    # Intersection.
    xx1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    yy1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    xx2 = np.minimum(a[:, 2][:, None], b[:, 2][None, :])
    yy2 = np.minimum(a[:, 3][:, None], b[:, 3][None, :])
    iw = np.clip(xx2 - xx1, 0.0, None)
    ih = np.clip(yy2 - yy1, 0.0, None)
    inter = iw * ih

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.where(union > 0, inter / union, 0.0)
    return ious


def iou_distance(atracks, btracks):
    """IoU cost (1 - IoU) between two lists of STrack (uses .tlbr)."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
            (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    _ious = iou_batch(atlbrs, btlbrs)
    return 1 - _ious
