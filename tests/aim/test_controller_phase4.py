"""Phase 4 AimController integration tests: trigger, recoil, target switching."""
from __future__ import annotations

from ragnarok.core.types import Track, Tracks, Team
from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.aim.velocity import VelocitySmoother
from ragnarok.aim.mouse import NullMouseDriver, MouseButton
from ragnarok.motion.shaper import NullShaper
from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator
from ragnarok.trigger.bot import TriggerBot
from ragnarok.tracking.egomotion import CommandedMotionBuffer


def _enemy(tid=1, xyxy=(250.0, 180.0, 290.0, 300.0)):   # to the RIGHT of the crosshair
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.ENEMY)


def _selector():
    return TargetSelector(fov_px=400.0, retain_fov_px=500.0, dwell_ms=0.0,
                          switch_margin=0.0, clock=lambda: 0)


def test_commanded_counts_pushed_to_buffer():
    cfg = AimConfig(enabled=True)
    buf = CommandedMotionBuffer()
    mouse = NullMouseDriver(); mouse.connect()
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
        shaper=NullShaper(), commanded_buffer=buf, clock=lambda: 1,
    )
    ac.update(Tracks((_enemy(),)), 0)
    # one (t, dcx, dcy) entry pushed this frame
    assert len(buf._buf) == 1


def test_trigger_fires_and_advances_recoil():
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    # crosshair is ROI centre (192,192); place an enemy whose box covers it.
    enemy = _enemy(xyxy=(150.0, 150.0, 240.0, 260.0))
    rc = RecoilCompensator(RecoilPattern(points=((0.0, 0.0), (0.0, 10.0))))
    trig = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
        shaper=NullShaper(), recoil=rc, trigger=trig,
        trigger_active=lambda: True, clock=lambda: 0,
    )
    ac.update(Tracks((enemy,)), 0)
    assert (MouseButton.LEFT, True) in mouse.buttons   # trigger fired
    assert rc._idx > 0                                 # recoil advanced on fire


def test_target_switch_releases_trigger():
    """_reset_stateful runs on target switch, releasing the trigger."""
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    # Two enemies both covering the crosshair (ROI centre = 192, 192).
    e1 = _enemy(tid=1, xyxy=(150.0, 150.0, 240.0, 260.0))
    e2 = _enemy(tid=2, xyxy=(150.0, 150.0, 240.0, 260.0))
    trig = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
        shaper=NullShaper(), trigger=trig, trigger_active=lambda: True,
        clock=lambda: 0,
    )
    # Frame 1: only e1 present → selected, trigger fires.
    ac.update(Tracks((e1,)), 0)
    assert (MouseButton.LEFT, True) in mouse.buttons
    # Frame 2: only e2 present → selector switches to tid=2 → _reset_stateful → release.
    ac.update(Tracks((e2,)), 8_000_000)
    assert (MouseButton.LEFT, False) in mouse.buttons


def test_disengage_releases_trigger():
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    enemy = _enemy(xyxy=(150.0, 150.0, 240.0, 260.0))
    trig = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    active = {"v": True}
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: active["v"], roi_size=384,
        shaper=NullShaper(), trigger=trig, trigger_active=lambda: True,
        clock=lambda: 0,
    )
    ac.update(Tracks((enemy,)), 0)
    active["v"] = False
    ac.update(Tracks((enemy,)), 8_000_000)
    assert (MouseButton.LEFT, False) in mouse.buttons   # released on disengage


def test_phase3_constructor_still_works():
    # Backward compat: no Phase 4 collaborators -> behaves like Phase 3.
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
    )
    for i in range(5):
        ac.update(Tracks((_enemy(),)), i * 8_000_000)
    assert sum(m[0] for m in mouse.moves) > 0           # still aims
    assert mouse.buttons == []                          # no trigger wired
