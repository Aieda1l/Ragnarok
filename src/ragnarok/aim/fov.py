"""FOV cone math — deg↔px conversions and aim-point helpers.

All arithmetic is pure (no IO, no side-effects) so it is directly unit-testable
without any GPU, display, or mouse.  The coordinate system is ROI-pixel space;
the crosshair lives at (roi_w/2, roi_h/2).

Phase 3: pixel space, identity ego-motion.  The deg↔px seam is isolated here
so Phase 4 (world-angular with real GMC) can be swapped without touching the
selector or aimer.
"""
from __future__ import annotations

import math

from ragnarok.core.types import Track


def focal_length_px(hfov_deg: float, screen_width_px: int) -> float:
    """Pinhole focal length (px) from the game's horizontal FOV.

    ``f = (screen_width_px / 2) / tan(hfov_deg / 2)``

    Parameters
    ----------
    hfov_deg:
        Game's full horizontal field-of-view in degrees.
    screen_width_px:
        Full render-target width in pixels (not the ROI crop width).
    """
    return (screen_width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def fov_deg_to_radius_px(
    aim_fov_deg: float,
    hfov_deg: float,
    screen_width_px: int,
) -> float:
    """Convert a half-angle aim cone to a screen-pixel radius.

    ``radius = focal_length_px * tan(aim_fov_deg / 2)``

    Because the focal length is a property of the *full* rendered screen, this
    radius is valid inside the ROI too (the crop doesn't change f).

    Parameters
    ----------
    aim_fov_deg:
        Full angle of the aim cone in degrees.
    hfov_deg:
        Game's full horizontal FOV in degrees.
    screen_width_px:
        Full render-target width in pixels.
    """
    f = focal_length_px(hfov_deg, screen_width_px)
    return f * math.tan(math.radians(aim_fov_deg) / 2.0)


def crosshair_for_roi(roi_w: int, roi_h: int) -> tuple[float, float]:
    """Return the crosshair position in ROI pixel coords (the ROI centre)."""
    return (roi_w / 2.0, roi_h / 2.0)


def dist_to(
    crosshair: tuple[float, float],
    pt: tuple[float, float],
) -> float:
    """Euclidean distance from *crosshair* to *pt* in pixel space."""
    return math.hypot(pt[0] - crosshair[0], pt[1] - crosshair[1])


def aim_point(
    tr: Track,
    head_frac: float = 0.15,
    mode: str = "head",
) -> tuple[float, float]:
    """Compute the 2-D aim point for *tr* in ROI pixel coords.

    Parameters
    ----------
    tr:
        The track whose bounding box is used.
    head_frac:
        Fraction of bbox height from the top for the ``"head"`` aim point.
        0.0 = top edge, 0.5 = vertical centre, 1.0 = bottom edge.
        Typical value ~0.12–0.18 for full-body detections.
    mode:
        ``"head"`` (default) positions the Y coordinate using *head_frac*;
        ``"body"`` returns the bbox centre regardless of *head_frac*.
    """
    x1, y1, x2, y2 = tr.xyxy
    cx = (x1 + x2) / 2.0
    if mode == "body":
        return (cx, (y1 + y2) / 2.0)
    # "head": head_frac from the top of the bounding box
    return (cx, y1 + head_frac * (y2 - y1))
