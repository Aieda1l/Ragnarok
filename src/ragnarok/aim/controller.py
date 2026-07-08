"""AimController — ties FOV/selection → IMM lead → aimer → shaper → mouse,
with feed-forward velocity, adaptive lead, recoil, and a safety-gated trigger.

All collaborators are injected; the Phase 4 ones are optional (default None) so
the Phase 3 constructor and tests keep working. Side effects (mouse move/button)
happen only while is_aim_active()/cfg.enabled (aim) and trigger_active() (fire).
Targets are ENEMY-only (the selector enforces this).

Pixel space this phase (identity ego-motion by default). Commanded counts are
pushed to a CommandedMotionBuffer so a FeedForwardGMC can back-project them.
"""
from __future__ import annotations

import math

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Team, Tracks
from ragnarok.aim.head import resolve_aim_point
from ragnarok.motion.shaper import NullShaper


def _clamp_between(a: float, b: float, v: float) -> float:
    """Clamp v to the closed interval [min(a,b), max(a,b)]."""
    lo, hi = (a, b) if a <= b else (b, a)
    return max(lo, min(hi, v))


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
        shaper=None,
        vel_smoother=None,
        adaptive_lead=None,
        recoil=None,
        trigger=None,
        trigger_active=None,
        line_clear=None,
        commanded_buffer=None,
    ) -> None:
        self._cfg = cfg
        self._sel = selector
        self._imm = imm_manager
        self._aimer = aimer
        self._mouse = mouse
        self._active = is_aim_active
        self._cx = roi_size / 2.0
        self._cy = roi_size / 2.0
        if getattr(cfg, "pinhole", False):
            focal = (cfg.screen_width_px / 2.0) / math.tan(math.radians(cfg.hfov_deg / 2.0))
            self._deg_per_px = math.degrees(1.0 / focal)   # true near-center rectilinear rate
        else:
            self._deg_per_px = cfg.hfov_deg / float(cfg.screen_width_px)
        self._deadtime_ns = round(float(getattr(cfg, "deadtime_ms", 0.0)) * 1e6)
        self._clock = clock
        self._shaper = shaper if shaper is not None else NullShaper()
        self._vel = vel_smoother
        self._lead = adaptive_lead
        self._recoil = recoil
        self._trigger = trigger
        self._trigger_active = trigger_active if trigger_active is not None else (lambda: False)
        self._line_clear = line_clear if line_clear is not None else (lambda: True)
        self._cmd_buf = commanded_buffer
        self._kff = float(getattr(cfg, "kff", 0.0))
        self._adaptive = bool(getattr(cfg, "adaptive_lead", False))
        self._last_ns: int | None = None
        self._last_shot_ns: int = 0            # last recoil-advance time (full-auto pacing)
        self._was_firing: bool = False         # trigger-release edge -> reset spray
        self._cur_target: int | None = None
        self.target_id: int | None = None          # aim lock (overlay / dynamic-ROI)
        self.fire_target_id: int | None = None      # ENEMY under the crosshair (trigger)
        self.aim_on: bool = False                    # live auto-aim toggle state (telemetry)
        self.trigger_on: bool = False                # live trigger toggle state (telemetry)

    def update(self, tracks: Tracks, t_capture_ns: int) -> None:
        self._imm.prune({t.track_id for t in tracks})

        # Read each toggle/key once per tick (a toggle closure is idempotent within
        # a tick, but reusing the value keeps the state and the gate consistent).
        aim_active = self._active()
        trig_active = self._trigger_active()
        self.aim_on = bool(self._cfg.enabled and aim_active)
        self.trigger_on = bool(self._trigger is not None and trig_active)

        # TRIGGER: evaluated every tick, independent of the aim key/toggle, so the
        # trigger bot fires on crosshair-over-enemy whether or not auto-aim is on.
        self._run_trigger(tracks, trig_active)

        # AIM ASSIST: only while aim is enabled AND its toggle/key is active.
        if not (self._cfg.enabled and aim_active):
            self._disengage_aim()
            return

        tid = self._sel.select(tracks, self._cx, self._cy)
        self.target_id = tid
        if tid is None:
            self._reset_aim_state()
            self._cur_target = None
            self._last_ns = t_capture_ns
            return

        if tid != self._cur_target:
            self._reset_aim_state()
            self._cur_target = tid

        track = next((t for t in tracks if t.track_id == tid), None)
        if track is None:
            return

        ax, ay = resolve_aim_point(
            track, tracks, mode=self._cfg.aim_point, head_frac=self._cfg.head_frac,
            head_class_id=getattr(self._cfg, "head_class_id", 1))
        dt = self._dt(t_capture_ns)
        self._imm.update(tid, ax, ay, dt)

        # predictive lead
        if self._lead is not None and self._adaptive:
            t_lead = self._lead.lead_seconds(t_capture_ns, self._clock())
        else:
            t_lead = self._cfg.lead_ms / 1000.0
        lead_pt = self._imm.lead(tid, t_lead)

        # feed-forward velocity (smoothed + clamped) — only if used
        tvx, tvy = 0.0, 0.0
        if self._kff > 0.0:
            vx, vy = self._imm.velocity(tid)
            if self._vel is not None:
                vx, vy = self._vel.smooth_clamp(vx, vy)
            tvx, tvy = vx, vy

        # Smith predictor / dead-time compensation: the aim path is a feedback
        # loop with a large round-trip delay (send -> game render -> display ->
        # capture -> detect -> track = tens of ms / several ticks). Advance the
        # crosshair by the counts already commanded but NOT yet visible in this
        # detection, so we don't re-issue a full correction every tick for moves
        # still in flight (that stacking is what makes every aimer overshoot and
        # rubber-band). Clamped so an over-estimated deadtime can't push the
        # crosshair past the target (never reverses the aim).
        chx, chy = self._cx, self._cy
        if self._cmd_buf is not None and self._deadtime_ns > 0 and self._deg_per_px > 0.0:
            now = self._clock()
            ccx, ccy = self._cmd_buf.integrate(now - self._deadtime_ns, now)
            inv_k = self._cfg.sensitivity / self._deg_per_px          # counts -> px
            chx = _clamp_between(self._cx, lead_pt[0], self._cx + ccx * inv_k)
            chy = _clamp_between(self._cy, lead_pt[1], self._cy + ccy * inv_k)

        dpx, dpy = self._aimer.step((chx, chy), lead_pt, dt, target_vel=(tvx, tvy))
        sx, sy = self._shaper.shape(dpx, dpy)

        k = self._deg_per_px / self._cfg.sensitivity   # px → mouse counts
        cdx, cdy = sx * k, sy * k
        self._mouse.move_relative(cdx, cdy)
        if self._cmd_buf is not None:
            self._cmd_buf.push(self._clock(), cdx, cdy)

    # ------------------------------------------------------------------
    # Trigger (independent of the aim key) + recoil
    # ------------------------------------------------------------------
    def _run_trigger(self, tracks: Tracks, active: bool) -> None:
        """Fire when the crosshair sits inside an ENEMY hitbox, gated only by the
        trigger's own activation. Runs every tick regardless of aim state, so the
        trigger bot works with auto-aim off. Recoil counter-moves are emitted here
        (as their own commanded move) so they also apply when aim is disengaged."""
        if self._trigger is None:
            self.fire_target_id = None
            return
        target = self._enemy_under_crosshair(tracks)
        self.fire_target_id = target.track_id if target is not None else None
        if target is None:
            self._trigger.release()
            if self._recoil is not None:
                self._recoil.release()
            self._was_firing = False
            return
        fired = self._trigger.update(
            track=target,
            crosshair=(self._cx, self._cy),
            occluded=target.time_since_update > 0,
            enemy_confirmed=True,           # crosshair-containment gate is ENEMY-only
            line_clear=self._line_clear(),
            active=active,
        )
        if self._recoil is not None:
            rx, ry = self._recoil_delta(fired)
            if rx or ry:
                k = self._deg_per_px / self._cfg.sensitivity
                cdx, cdy = rx * k, ry * k
                self._mouse.move_relative(cdx, cdy)
                if self._cmd_buf is not None:
                    self._cmd_buf.push(self._clock(), cdx, cdy)

    def _recoil_delta(self, fired: bool) -> tuple[float, float]:
        """Per-tick recoil counter-move (px). A new press advances one shot; while
        HELD with fire_rate_rps > 0 (full-auto) it advances one shot every 1/rps s;
        rps == 0 stays semi-auto. Resets the spray on the fire-release edge."""
        now = self._clock()
        rps = getattr(self._recoil, "fire_rate_rps", 0.0)
        firing = self._trigger.is_firing
        rx = ry = 0.0
        if self._was_firing and not firing:                # released -> reset spray
            self._recoil.release()
        if fired:                                          # a shot fired
            rx, ry = self._recoil.on_fire()
            self._last_shot_ns = now
        elif rps > 0.0 and firing and now - self._last_shot_ns >= 1e9 / rps:
            rx, ry = self._recoil.on_fire()                # held full-auto -> next shot
            self._last_shot_ns = now
        self._was_firing = firing
        return rx, ry

    def _enemy_under_crosshair(self, tracks: Tracks):
        for t in tracks:
            if t.team is Team.ENEMY:
                x1, y1, x2, y2 = t.xyxy
                if x1 <= self._cx <= x2 and y1 <= self._cy <= y2:
                    return t
        return None

    def _dt(self, t_ns: int) -> float:
        if self._last_ns is None:
            self._last_ns = t_ns
            return 1.0 / 120.0
        dt = (t_ns - self._last_ns) / 1e9
        self._last_ns = t_ns
        return max(1e-3, min(0.1, dt))

    def _reset_aim_state(self) -> None:
        """Reset aim-assist state only (aimer/shaper/velocity). The trigger and
        recoil are managed independently in _run_trigger, so a target switch or an
        aim disengage never releases the trigger."""
        if self._aimer is not None:
            self._aimer.reset()
        self._shaper.reset()
        if self._vel is not None:
            self._vel.reset()

    def _disengage_aim(self) -> None:
        self._reset_aim_state()
        self._sel.reset()
        self._last_ns = None
        self._cur_target = None
        self.target_id = None

    def release(self) -> None:
        """Release any held trigger button (and reset the spray). Called on hot-swap
        so a held fire doesn't stick when this controller is replaced. Does NOT close
        the mouse driver — that is owned and shared by the app."""
        if self._trigger is not None:
            self._trigger.release()
        if self._recoil is not None:
            self._recoil.release()
