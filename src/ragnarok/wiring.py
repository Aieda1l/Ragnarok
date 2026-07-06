"""Config -> concrete component builders (the live-wiring follow-up).

Keeps the heavy/optional imports (vendored BoT-SORT, OpenCV HSV classifier) out
of ``app.py`` and importable/testable without PySide6 or a GPU. ``app.py`` calls
these to turn a frozen ``AppConfig`` into the real ``Tracker`` and friend/foe
``FriendFoeClassifier`` that make the aim core engage against the live game.
"""
from __future__ import annotations

from ragnarok.config.schema import AppConfig
from ragnarok.tracking.base import Tracker, IdentityTracker
from ragnarok.classification.base import (
    FriendFoeClassifier, NullClassifier, AllEnemyClassifier)


def build_tracker(cfg: AppConfig, *, gmc_buffer=None) -> Tracker:
    t = cfg.tracking
    if t.backend == "identity":
        return IdentityTracker()
    from ragnarok.tracking.botsort import BotSortTracker
    ego = None
    if t.gmc == "feedforward" and gmc_buffer is not None:
        from ragnarok.tracking.egomotion import FeedForwardGMC
        ego = FeedForwardGMC(
            hfov_deg=cfg.aim.hfov_deg, screen_width_px=cfg.aim.screen_width_px,
            deg_per_count=t.deg_per_count, tau_render_s=t.tau_render_s,
            frame_dt_s=1.0 / cfg.capture.target_fps, buffer=gmc_buffer,
        )
    return BotSortTracker(
        ego=ego,                                  # None -> BotSortTracker uses IdentityEgoMotion
        track_high_thresh=t.track_high_thresh,
        track_low_thresh=t.track_low_thresh,
        new_track_thresh=t.new_track_thresh,
        track_buffer=t.track_buffer,
        match_thresh=t.match_thresh,
        proximity_thresh=t.proximity_thresh,
        frame_rate=cfg.capture.target_fps,
    )


def build_classifier(cfg: AppConfig) -> FriendFoeClassifier:
    c = cfg.classification
    if not c.enabled:
        return AllEnemyClassifier()          # off => treat every detection as a target
    from ragnarok.classification.base import HSVRingClassifier
    from ragnarok.classification.color import resolve_enemy_profile
    profile = resolve_enemy_profile(c.palette, c.enemy_color)
    return HSVRingClassifier(
        profile,
        frac_threshold=c.frac_threshold,
        thickness=c.thickness,
        vote_window=c.vote_window,
        vote_min=c.vote_min,
    )


def build_aimer(cfg: AppConfig):
    a = cfg.aim
    from ragnarok.aim.aimers import (
        FlickAimer, FeedbackAimer, HybridAimer, PredictiveAimer,
    )
    if a.aimer == "flick":
        return FlickAimer(flick_speed_px_s=a.flick_speed_px_s)
    if a.aimer == "hybrid":
        return HybridAimer(
            kp=a.kp, max_step_px=a.max_step_px,
            flick_dist_px=a.hybrid_flick_dist_px,
            flick_speed_px_s=a.flick_speed_px_s, ema_alpha=a.ema_alpha,
        )
    if a.aimer == "predictive":
        return PredictiveAimer(max_step_px=a.max_step_px, kff=a.kff)
    ki = a.ki if a.controller_mode in ("pi", "pid") else 0.0
    kd = a.kd if a.controller_mode == "pid" else 0.0
    return FeedbackAimer(
        kp=a.kp, max_step_px=a.max_step_px, ema_alpha=a.ema_alpha, kff=a.kff,
        ki=ki, kd=kd, integral_clamp=a.integral_clamp,
        cond_integ_thresh_px=a.cond_integ_thresh_px, creep_px=a.creep_px,
    )


def build_shaper(cfg: AppConfig):
    m = cfg.motion
    from ragnarok.motion.shaper import NullShaper, WindMouseShaper
    if m.shaper == "windmouse":
        return WindMouseShaper(
            gravity=m.gravity, wind=m.wind,
            max_step=m.max_step, target_area=m.target_area,
        )
    return NullShaper()


def build_recoil(cfg: AppConfig):
    r = cfg.recoil
    if not r.enabled or not r.pattern:
        return None
    from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator
    return RecoilCompensator(RecoilPattern(points=r.pattern), scale=r.scale,
                             fire_rate_rps=r.fire_rate_rps)


def build_mouse_driver(cfg, *, sendinput_factory, arduino_factory):
    """Select the mouse driver from config. Selection-only + import-light: the
    real SendInput/Arduino builds live in the injected factories (box-only)."""
    if cfg.input.mouse_driver == "arduino":
        return arduino_factory(cfg)
    return sendinput_factory()
