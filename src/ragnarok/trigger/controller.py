"""Standalone trigger controller (spec §6.7) — fire without aim assist.

Occupies the worker loop's per-tick slot (same ``.update(tracks, t_ns)`` +
``target_id`` surface as AimController) when aim is disabled but the trigger bot
is enabled. Fires when the crosshair sits inside an ENEMY hitbox and every
TriggerBot gate holds; on each new shot it advances + applies the recoil
counter-move (px → mouse counts), exactly as the aim-coupled path does. Built
mutually exclusive with AimController, so the trigger never double-fires.
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Team, Tracks


class TriggerController:
    def __init__(
        self,
        cfg,                       # AimConfig — hfov/screen_width/sensitivity for px→counts
        *,
        trigger,
        trigger_active,
        mouse,
        roi_size: int,
        recoil=None,
        line_clear=None,
        clock=now_ns,
    ) -> None:
        self._cfg = cfg
        self._trigger = trigger
        self._trigger_active = trigger_active if trigger_active is not None else (lambda: False)
        self._mouse = mouse
        self._recoil = recoil
        self._line_clear = line_clear if line_clear is not None else (lambda: True)
        self._cx = roi_size / 2.0
        self._cy = roi_size / 2.0
        self._k = (cfg.hfov_deg / float(cfg.screen_width_px)) / cfg.sensitivity
        self._clock = clock
        self.target_id: int | None = None

    def update(self, tracks: Tracks, t_capture_ns: int) -> None:
        target = self._enemy_under_crosshair(tracks)
        self.target_id = target.track_id if target is not None else None
        if target is None:
            self._trigger.release()
            if self._recoil is not None:
                self._recoil.release()
            return
        fired = self._trigger.update(
            track=target,
            crosshair=(self._cx, self._cy),
            occluded=target.time_since_update > 0,
            enemy_confirmed=True,               # crosshair-containment gate below is ENEMY-only
            line_clear=self._line_clear(),
            active=self._trigger_active(),
        )
        if fired and self._recoil is not None:
            rx, ry = self._recoil.on_fire()
            self._mouse.move_relative(rx * self._k, ry * self._k)

    def _enemy_under_crosshair(self, tracks):
        for t in tracks:
            if t.team is Team.ENEMY:
                x1, y1, x2, y2 = t.xyxy
                if x1 <= self._cx <= x2 and y1 <= self._cy <= y2:
                    return t
        return None
