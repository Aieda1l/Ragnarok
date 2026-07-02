"""Pure geometry for the smart-lock FOV overlay (spec §10.2).

ZERO Qt dependency: turns a telemetry snapshot + config into a set of draw
primitives (an ``OverlayScene``) in absolute screen coordinates. The QPainter
widget (``overlay_window.py``) consumes this; all the math lives here so it is
unit-testable without any display, GPU, or mouse.

Coordinates: tracks live in ROI-pixel space (crosshair at ROI centre).
``ScreenMap`` maps ROI px -> absolute screen px using the captured region.
FOV radii come from ``aim.fov.fov_deg_to_radius_px`` (screen px). The normal
capture path is 1:1 (region width == roi_size, scale 1.0); dynamic-ROI tracks
are already mapped back to full-frame coords upstream, so scale stays 1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ragnarok.core.types import Team, Track
from ragnarok.aim.fov import aim_point, fov_deg_to_radius_px


@dataclass(frozen=True)
class ScreenMap:
    left: float
    top: float
    scale_x: float
    scale_y: float

    @classmethod
    def from_region(cls, region, roi_w: int, roi_h: int) -> "ScreenMap":
        left, top, right, bottom = region
        return cls(left=float(left), top=float(top),
                   scale_x=(right - left) / float(roi_w),
                   scale_y=(bottom - top) / float(roi_h))

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return (self.left + x * self.scale_x, self.top + y * self.scale_y)

    def rect(self, xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        sx1, sy1 = self.pt(x1, y1)
        sx2, sy2 = self.pt(x2, y2)
        return (sx1, sy1, sx2, sy2)


@dataclass(frozen=True)
class FovRing:
    center: tuple[float, float]
    acquire_radius: float     # inner (acquisition)
    retain_radius: float      # outer (sticky retention)
    tick_count: int = 12


@dataclass(frozen=True)
class TargetMarker:
    track_id: int
    box: tuple[float, float, float, float]   # screen xyxy
    diamond: tuple[float, float]             # screen aim-point (confirmed-target marker)
    team: Team
    confidence: float
    locked: bool
    in_fov: bool


@dataclass(frozen=True)
class OffscreenHint:
    angle_rad: float                         # direction from crosshair
    edge_point: tuple[float, float]          # clamped to the viewport edge
    team: Team


@dataclass(frozen=True)
class OverlayScene:
    has_signal: bool
    crosshair: tuple[float, float]
    fov: FovRing | None
    markers: tuple[TargetMarker, ...]
    bracket_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    locked_line: tuple[tuple[float, float], tuple[float, float]] | None
    offscreen: tuple[OffscreenHint, ...]

    @classmethod
    def empty(cls) -> "OverlayScene":
        return cls(has_signal=False, crosshair=(0.0, 0.0), fov=None, markers=(),
                   bracket_segments=(), locked_line=None, offscreen=())
