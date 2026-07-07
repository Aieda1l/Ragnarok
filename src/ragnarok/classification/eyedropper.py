"""Eyedropper: build a custom enemy HSV band from a sampled pixel (spec §11).

Lets the user match an in-game enemy outline colour that isn't one of the built-in
palettes by sampling it off a frame. Pure/testable: the BGR->HSV band math and the
band->ColorProfile build. The click-on-preview capture is box-only (GUI panel).
"""
from __future__ import annotations

import cv2
import numpy as np

from ragnarok.classification.color import ColorProfile, HSVBand


def hsv_band_from_bgr(bgr, *, h_tol: int = 10, s_tol: int = 70, v_tol: int = 70):
    """Sampled BGR pixel -> a 6-int HSV band (h_lo,h_hi,s_lo,s_hi,v_lo,v_hi) on
    OpenCV scales (H 0-179, S/V 0-255), centred on the pixel's hue with S/V open
    to the bright end (enemy outlines are saturated glows)."""
    px = np.array([[list(bgr)]], dtype=np.uint8)          # 1x1 BGR
    h, s, v = (int(x) for x in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0])
    return (max(0, h - h_tol), min(179, h + h_tol),
            max(0, s - s_tol), 255,
            max(0, v - v_tol), 255)


def profile_from_band(band) -> ColorProfile:
    """A single-band ``ColorProfile`` from a 6-int band tuple."""
    return ColorProfile("custom", (HSVBand(*band),))
