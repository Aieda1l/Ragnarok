"""Smart-sampling capture decision for the frame grabber (spec §12 step 1).

Pure functions: keep a frame for labeling when the detector is UNCERTAIN about
it (no/low-confidence detections — the hard examples worth labeling) or when the
SCENE CHANGED vs the last saved frame (coverage/diversity). No IO here.
"""
from __future__ import annotations

import numpy as np


def scene_change_fraction(img_a, img_b) -> float:
    a = np.asarray(img_a)
    b = np.asarray(img_b)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)


def should_capture(detections, frame_image, last_saved_image, *,
                   conf_threshold: float, scene_change_threshold: float) -> bool:
    confs = [d.confidence for d in detections]
    uncertain = (not confs) or (max(confs) < conf_threshold)
    if uncertain:
        return True
    if last_saved_image is None:
        return True
    return scene_change_fraction(frame_image, last_saved_image) >= scene_change_threshold
