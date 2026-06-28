"""HSV outline-ring color matching for friend/foe classification.

Pure numpy + OpenCV; unit-testable on synthetic images (no display).
OpenCV HSV scale: H in 0-179, S/V in 0-255. RED straddles H=0 so it needs
two bands OR'd. Keep S/V floors high to reject desaturated HUD/shadows.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class HSVBand:
    """One cv2.inRange band on OpenCV scales (H 0-179, S/V 0-255)."""
    h_lo: int
    h_hi: int
    s_lo: int = 110
    s_hi: int = 255
    v_lo: int = 110
    v_hi: int = 255

    def lower(self) -> np.ndarray:
        return np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8)

    def upper(self) -> np.ndarray:
        return np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8)


@dataclass(frozen=True)
class ColorProfile:
    name: str
    bands: tuple[HSVBand, ...]  # >1 band => OR'd (RED wraparound)


# --- Vivid in-game default enemy outline colors (saturated glow) ---
RED = ColorProfile("red", (
    HSVBand(0, 10, 110, 255, 110, 255),     # low-hue side of the wrap
    HSVBand(170, 179, 110, 255, 110, 255),  # high-hue side of the wrap
))
PURPLE = ColorProfile("purple", (HSVBand(120, 155, 80, 255, 80, 255),))
YELLOW = ColorProfile("yellow", (HSVBand(22, 38, 110, 255, 110, 255),))
DEFAULT_ENEMY_PROFILES: dict[str, ColorProfile] = {
    "red": RED, "purple": PURPLE, "yellow": YELLOW,
}

# --- Wong / Okabe-Ito colorblind-safe selectable presets (approx HSV bands) ---
WONG_PROFILES: dict[str, ColorProfile] = {
    "orange":         ColorProfile("orange",         (HSVBand(13, 27, 120, 255, 120, 255),)),
    "sky_blue":       ColorProfile("sky_blue",       (HSVBand(92, 108, 90, 255, 120, 255),)),
    "bluish_green":   ColorProfile("bluish_green",   (HSVBand(74, 92, 120, 255, 90, 255),)),
    "yellow":         ColorProfile("yellow",         (HSVBand(22, 36, 120, 255, 120, 255),)),
    "blue":           ColorProfile("blue",           (HSVBand(95, 112, 130, 255, 110, 255),)),
    "vermillion":     ColorProfile("vermillion",     (HSVBand(6, 20, 130, 255, 120, 255),)),
    "reddish_purple": ColorProfile("reddish_purple", (HSVBand(150, 172, 70, 220, 120, 255),)),
}


# Selectable palettes: the vivid in-game defaults and the colorblind-safe set.
PALETTES: dict[str, dict[str, ColorProfile]] = {
    "default": DEFAULT_ENEMY_PROFILES,
    "wong": WONG_PROFILES,
}


def resolve_enemy_profile(palette: str, color: str) -> ColorProfile:
    """Look up a ColorProfile by palette + color name.

    Raises ValueError (listing valid choices) on an unknown palette or color, so
    a config typo fails fast at startup rather than silently never matching."""
    try:
        table = PALETTES[palette]
    except KeyError:
        raise ValueError(
            f"unknown palette {palette!r}; choose from {sorted(PALETTES)}"
        ) from None
    try:
        return table[color]
    except KeyError:
        raise ValueError(
            f"unknown enemy_color {color!r} for palette {palette!r}; "
            f"choose from {sorted(table)}"
        ) from None


def ring_mask(shape_hw: tuple[int, int], xyxy, thickness: int = 4) -> np.ndarray:
    """Boolean annulus straddling the bbox border (where the outline glow sits).

    shape_hw = (H, W). Returns a boolean array; the box interior and far
    background are False. A zero-area box yields an all-False mask.
    """
    h, w = shape_hw
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    box = np.zeros((h, w), np.uint8)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(box, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * thickness + 1, 2 * thickness + 1))
    outer = cv2.dilate(box, k, iterations=1)
    inner = cv2.erode(box, k, iterations=1)
    return cv2.subtract(outer, inner).astype(bool)


def color_match_fraction(img_bgr: np.ndarray, xyxy, profile: ColorProfile,
                         thickness: int = 4, open_ksize: int = 3) -> float:
    """Fraction of ring pixels matching the enemy color profile (0..1)."""
    h, w = img_bgr.shape[:2]
    ring = ring_mask((h, w), xyxy, thickness)
    n_ring = int(ring.sum())
    if n_ring == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    color = np.zeros((h, w), np.uint8)
    for b in profile.bands:                       # OR bands (RED wraparound)
        color = cv2.bitwise_or(color, cv2.inRange(hsv, b.lower(), b.upper()))
    if open_ksize >= 3:                           # drop speckle
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        color = cv2.morphologyEx(color, cv2.MORPH_OPEN, ok)
    matched = np.logical_and(color > 0, ring)
    return int(matched.sum()) / float(n_ring)


def is_enemy_frame(img_bgr: np.ndarray, xyxy, profile: ColorProfile,
                   frac_threshold: float = 0.18, **kw) -> bool:
    return color_match_fraction(img_bgr, xyxy, profile, **kw) >= frac_threshold
