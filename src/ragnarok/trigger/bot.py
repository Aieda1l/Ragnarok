"""Trigger bot with safety gates (spec §6.7).

Fires ONLY when every gate holds continuously for activation_delay_s:
  active (trigger key) AND enemy_confirmed AND not occluded (never a coasted
  box) AND line_clear (no teammate pixel on the path) AND crosshair inside the
  hitbox. Any gate dropping releases the button immediately. update() returns
  True on the frame a NEW press is issued, so the caller advances recoil.
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.aim.mouse import MouseButton


class TriggerBot:
    def __init__(
        self,
        *,
        mouse,
        activation_delay_s: float,
        button: MouseButton = MouseButton.LEFT,
        clock=now_ns,
    ) -> None:
        self._mouse = mouse
        self._delay = activation_delay_s
        self._button = button
        self._clock = clock
        self._pressed = False
        self._eligible_since: int | None = None

    @staticmethod
    def _inside(track, crosshair) -> bool:
        x1, y1, x2, y2 = track.xyxy
        cx, cy = crosshair
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def update(
        self,
        *,
        track,
        crosshair,
        occluded: bool,
        enemy_confirmed: bool,
        line_clear: bool,
        active: bool,
    ) -> bool:
        ready = (
            active
            and enemy_confirmed
            and not occluded
            and line_clear
            and track is not None
            and self._inside(track, crosshair)
        )
        if not ready:
            self._eligible_since = None
            self._release_if_pressed()
            return False

        now = self._clock()
        if self._eligible_since is None:
            self._eligible_since = now
        elapsed = (now - self._eligible_since) / 1e9
        if elapsed >= self._delay and not self._pressed:
            self._mouse.set_button(self._button, True)
            self._pressed = True
            return True
        return False

    def release(self) -> None:
        self._eligible_since = None
        self._release_if_pressed()

    def _release_if_pressed(self) -> None:
        if self._pressed:
            self._mouse.set_button(self._button, False)
            self._pressed = False
