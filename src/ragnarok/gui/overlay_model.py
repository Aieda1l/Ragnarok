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


class LockAgeTracker:
    """Tracks how long the current lock has been held, for the lock-on
    convergence animation. Clock-agnostic: the caller passes ``now_ns`` each
    frame (widget uses ``core.clock.now_ns``; tests pass explicit values)."""

    def __init__(self) -> None:
        self._locked: int | None = None
        self._start_ns: int | None = None

    def update(self, locked_id: int | None, now_ns: int) -> float:
        if locked_id != self._locked:
            self._locked = locked_id
            self._start_ns = now_ns if locked_id is not None else None
        if self._start_ns is None:
            return 0.0
        return (now_ns - self._start_ns) / 1e9


def lock_progress(lock_age_s: float, duration_s: float) -> float:
    """Convergence parameter for the lock-on brackets: 0 (wide) -> 1 (snapped)."""
    if duration_s <= 0.0:
        return 1.0
    return max(0.0, min(1.0, lock_age_s / duration_s))


def bracket_segments(box, t: float, gap: float, arm_len: float):
    """L-shaped corner brackets that converge onto ``box`` as ``t`` goes 0->1.

    ``t=0``: each corner sits ``gap`` px outside the box (unconverged/wide).
    ``t=1``: each corner sits exactly on the box corner (snapped/locked).
    Returns 8 segments (horizontal + vertical arm per corner), each a pair of
    screen points ``((x1,y1),(x2,y2))``. Arms always point *inward*.
    """
    x1, y1, x2, y2 = box
    o = gap * (1.0 - t)                      # outward offset shrinks to 0
    L = arm_len
    corners = (
        (x1 - o, y1 - o, +1, +1),           # top-left    -> arms right & down
        (x2 + o, y1 - o, -1, +1),           # top-right   -> arms left & down
        (x1 - o, y2 + o, +1, -1),           # bottom-left -> arms right & up
        (x2 + o, y2 + o, -1, -1),           # bottom-right-> arms left & up
    )
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for cx, cy, sx, sy in corners:
        segs.append(((cx, cy), (cx + sx * L, cy)))   # horizontal arm
        segs.append(((cx, cy), (cx, cy + sy * L)))   # vertical arm
    return tuple(segs)


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


def build_scene(*, snapshot, cfg, viewport, lock_age_s: float,
                bracket_gap: float = 28.0, bracket_arm: float = 16.0,
                bracket_anim_s: float = 0.18) -> OverlayScene:
    """Assemble the full ``OverlayScene`` from a telemetry snapshot + config.

    Returns an empty (no-signal) scene when the snapshot has no ROI region.
    ``viewport`` is the overlay's screen rect ``(x0,y0,x1,y1)`` used for the
    off-screen direction hints. ``lock_age_s`` drives the bracket convergence.
    """
    region = snapshot.roi_region
    if region is None:
        return OverlayScene.empty()

    roi = cfg.capture.roi_size
    smap = ScreenMap.from_region(region, roi, roi)
    crosshair = smap.pt(roi / 2.0, roi / 2.0)

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    fov = FovRing(center=crosshair, acquire_radius=fov_px, retain_radius=retain_px)

    markers = build_markers(snapshot.tracks, smap, crosshair, fov_px,
                            snapshot.locked_target_id, a.head_frac, a.aim_point)

    segs: tuple = ()
    line = None
    locked = next((m for m in markers if m.locked), None)
    if locked is not None:
        t = lock_progress(lock_age_s, bracket_anim_s)
        segs = bracket_segments(locked.box, t, bracket_gap, bracket_arm)
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
                        markers=markers, bracket_segments=segs,
                        locked_line=line, offscreen=tuple(hints))
