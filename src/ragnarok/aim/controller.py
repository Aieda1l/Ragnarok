"""AimController — ties FOV/selection → IMM lead → aimer → mouse each frame.

Runs after classify in the worker loop. All collaborators are injected so the
controller is fully unit-testable with a NullMouseDriver + fake key provider
(no real cursor/keys). Side effects happen only while ``is_aim_active()`` is
true and ``cfg.enabled``. Targets are ENEMY-only (enforced by the selector).

Phase 3 = pixel space (identity ego-motion): the crosshair-relative pixel error
maps to mouse counts by ``deg_per_px / sensitivity``. World-angular is Phase 4.
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Tracks
from ragnarok.aim.fov import aim_point


class AimController:
    def __init__(
        self,
        cfg,
        *,
        selector,
        imm_manager,
        aimer,
        mouse,
        is_aim_active,
        roi_size: int,
        clock=now_ns,
    ) -> None:
        self._cfg = cfg
        self._sel = selector
        self._imm = imm_manager
        self._aimer = aimer
        self._mouse = mouse
        self._active = is_aim_active
        self._cx = roi_size / 2.0
        self._cy = roi_size / 2.0
        # degrees subtended per screen pixel (identity ego-motion this phase)
        self._deg_per_px = cfg.hfov_deg / float(cfg.screen_width_px)
        self._clock = clock
        self._last_ns: int | None = None
        self._cur_target: int | None = None
        self.target_id: int | None = None

    def update(self, tracks: Tracks, t_capture_ns: int) -> None:
        self._imm.prune({t.track_id for t in tracks})

        if not (self._cfg.enabled and self._active()):
            self._disengage()
            return

        tid = self._sel.select(tracks, self._cx, self._cy)
        self.target_id = tid
        if tid is None:
            self._aimer.reset()
            self._cur_target = None
            self._last_ns = t_capture_ns
            return

        if tid != self._cur_target:        # target switch → re-latch flick / clear EMA
            self._aimer.reset()
            self._cur_target = tid

        track = next((t for t in tracks if t.track_id == tid), None)
        if track is None:
            return

        ax, ay = aim_point(track, self._cfg.head_frac, self._cfg.aim_point)
        dt = self._dt(t_capture_ns)
        self._imm.update(tid, ax, ay, dt)
        lead_pt = self._imm.lead(tid, self._cfg.lead_ms / 1000.0)
        dpx, dpy = self._aimer.step((self._cx, self._cy), lead_pt, dt)
        k = self._deg_per_px / self._cfg.sensitivity   # px → mouse counts
        self._mouse.move_relative(dpx * k, dpy * k)

    def _dt(self, t_ns: int) -> float:
        if self._last_ns is None:
            self._last_ns = t_ns
            return 1.0 / 120.0
        dt = (t_ns - self._last_ns) / 1e9
        self._last_ns = t_ns
        return max(1e-3, min(0.1, dt))

    def _disengage(self) -> None:
        self._aimer.reset()
        self._sel.reset()
        self._last_ns = None
        self._cur_target = None
        self.target_id = None
