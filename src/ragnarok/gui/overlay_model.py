"""Pure geometry for the smart-weapon FOV overlay (spec §10.2).

Modeled on the Cyberpunk 2077 smart-link reticle: a square FOV framed by two
brackets (a thin vertical line with two bold ~45° diagonal arms at its top and
bottom), and red diamonds marking detected aim points.

ZERO Qt dependency: turns a telemetry snapshot + config into a set of draw
primitives (an ``OverlayScene``) in absolute screen coordinates. The QPainter
widget (``overlay_window.py``) consumes this; all the math lives here so it is
unit-testable without any display, GPU, or mouse.

Coordinates: tracks live in ROI-pixel space (crosshair at ROI centre).
``ScreenMap`` maps ROI px -> absolute screen px using the captured region.
The FOV half-extent comes from ``aim.fov.fov_deg_to_radius_px`` (screen px). The
normal capture path is 1:1 (region width == roi_size, scale 1.0); dynamic-ROI
tracks are already mapped back to full-frame coords upstream, so scale stays 1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ragnarok.core.types import Team, Track
from ragnarok.aim.fov import aim_point, fov_deg_to_radius_px

# A line segment is a pair of screen points ((x1,y1),(x2,y2)).
_Seg = tuple[tuple[float, float], tuple[float, float]]


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
class FovBox:
    """The square FOV lock-zone: corners at ``center ± half`` on each axis."""
    center: tuple[float, float]
    half: float


@dataclass(frozen=True)
class TargetMarker:
    track_id: int
    box: tuple[float, float, float, float]   # screen xyxy
    diamond: tuple[float, float]             # screen aim-point (red diamond)
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
    fov: FovBox | None
    fov_thin: tuple[_Seg, ...]               # the two thin vertical bracket lines
    fov_thick: tuple[_Seg, ...]              # the four bold 45° corner arms
    markers: tuple[TargetMarker, ...]
    locked_line: _Seg | None                 # crosshair -> locked diamond
    offscreen: tuple[OffscreenHint, ...]

    @classmethod
    def empty(cls) -> "OverlayScene":
        return cls(has_signal=False, crosshair=(0.0, 0.0), fov=None,
                   fov_thin=(), fov_thick=(), markers=(), locked_line=None,
                   offscreen=())


def fov_bracket_segments(center, half: float, arm: float):
    """Two smart-weapon brackets framing a square FOV of half-extent ``half``.

    Returns ``(thin, thick)``:
      * ``thin`` — the two vertical bracket lines (drawn thin), at ``cx ± half``.
      * ``thick`` — the four bold diagonal arms (~45°) at the top and bottom of
        each vertical, angling INWARD toward the FOV centre.

    A single arm is a true 45° diagonal (equal dx/dy of magnitude ``arm``).
    """
    cx, cy = center
    left, right = cx - half, cx + half
    top, bot = cy - half, cy + half
    thin: tuple[_Seg, ...] = (
        ((left, top), (left, bot)),      # left vertical
        ((right, top), (right, bot)),    # right vertical
    )
    # bold arms flare OUTWARD and UP/DOWN from each vertical's ends (the top arm
    # points up, the bottom arm points down) — they do NOT angle inward toward
    # the aim point.
    thick: tuple[_Seg, ...] = (
        ((left, top), (left - arm, top - arm)),      # left-top   -> up & out (up-left)
        ((left, bot), (left - arm, bot + arm)),      # left-bottom-> down & out (down-left)
        ((right, top), (right + arm, top - arm)),    # right-top  -> up & out (up-right)
        ((right, bot), (right + arm, bot + arm)),    # right-bottom-> down & out (down-right)
    )
    return thin, thick


def build_markers(tracks, smap: ScreenMap, crosshair: tuple[float, float],
                  fov_px: float, locked_id: int | None,
                  head_frac: float, aim_mode: str) -> tuple[TargetMarker, ...]:
    """One ``TargetMarker`` per track, in screen coords.

    ``in_fov`` compares the screen-space crosshair->aim-point distance to
    ``fov_px`` (both screen px; scale is 1.0 on the normal capture path).
    """
    out: list[TargetMarker] = []
    for tr in tracks:
        apx, apy = aim_point(tr, head_frac, aim_mode)
        diamond = smap.pt(apx, apy)
        d = math.hypot(diamond[0] - crosshair[0], diamond[1] - crosshair[1])
        out.append(TargetMarker(
            track_id=tr.track_id, box=smap.rect(tr.xyxy), diamond=diamond,
            team=tr.team, confidence=tr.confidence,
            locked=(locked_id is not None and tr.track_id == locked_id),
            in_fov=(d <= fov_px),
        ))
    return tuple(out)


def _in_viewport(pt, viewport) -> bool:
    x, y = pt
    x0, y0, x1, y1 = viewport
    return x0 <= x <= x1 and y0 <= y <= y1


def _ray_rect_edge(origin, target, viewport):
    """Point where ray origin->target first crosses the viewport rectangle.

    Returns ``target`` unchanged if the ray is degenerate / never hits (should
    not happen for an off-screen target with an in-viewport origin).
    """
    ox, oy = origin
    dx, dy = target[0] - ox, target[1] - oy
    x0, y0, x1, y1 = viewport
    eps = 1e-6
    best: float | None = None
    for denom, num in ((dx, x0 - ox), (dx, x1 - ox), (dy, y0 - oy), (dy, y1 - oy)):
        if denom == 0.0:
            continue
        s = num / denom
        if s <= 0.0:
            continue
        px, py = ox + dx * s, oy + dy * s
        if (x0 - eps <= px <= x1 + eps) and (y0 - eps <= py <= y1 + eps):
            if best is None or s < best:
                best = s
    if best is None:
        return target
    return (ox + dx * best, oy + dy * best)


def build_scene(*, snapshot, cfg, viewport, bracket_arm: float = 10.0) -> OverlayScene:
    """Assemble the full ``OverlayScene`` from a telemetry snapshot + config.

    Returns an empty (no-signal) scene when the snapshot has no ROI region.
    The FOV square is framed by two smart-weapon brackets sized from the acquire
    cone; each ENEMY aim point gets a red diamond (the locked one is emphasised),
    with a thin tracking line to the locked target. ``viewport`` is the overlay's
    screen rect ``(x0,y0,x1,y1)`` used for the off-screen direction hints.
    """
    region = snapshot.roi_region
    if region is None:
        return OverlayScene.empty()

    roi = cfg.capture.roi_size
    smap = ScreenMap.from_region(region, roi, roi)
    crosshair = smap.pt(roi / 2.0, roi / 2.0)

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    fov = FovBox(center=crosshair, half=fov_px)
    thin, thick = fov_bracket_segments(crosshair, fov_px, min(bracket_arm, fov_px))

    markers = build_markers(snapshot.tracks, smap, crosshair, fov_px,
                            snapshot.locked_target_id, a.head_frac, a.aim_point)

    line: _Seg | None = None
    locked = next((m for m in markers if m.locked), None)
    if locked is not None:
        line = (crosshair, locked.diamond)

    # Off-screen direction hints. NOTE: with the current single centered-ROI
    # capture path every detected target is inside the ROI and therefore inside
    # the viewport, so this stays empty at runtime; it fires once an off-screen
    # target source exists (full-frame detection, wider capture, or coasted
    # tracks that leave the ROI). The geometry is exercised by the unit tests.
    hints: list[OffscreenHint] = []
    for m in markers:
        if m.team is Team.ENEMY and not _in_viewport(m.diamond, viewport):
            edge = _ray_rect_edge(crosshair, m.diamond, viewport)
            ang = math.atan2(m.diamond[1] - crosshair[1], m.diamond[0] - crosshair[0])
            hints.append(OffscreenHint(angle_rad=ang, edge_point=edge, team=m.team))

    return OverlayScene(has_signal=True, crosshair=crosshair, fov=fov,
                        fov_thin=thin, fov_thick=thick, markers=markers,
                        locked_line=line, offscreen=tuple(hints))
