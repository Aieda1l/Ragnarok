from ragnarok.core.types import Track, Tracks, Team
from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import FeedbackAimer, Aimer
from ragnarok.aim.mouse import NullMouseDriver


def _enemy(tid=1, xyxy=(250.0, 180.0, 290.0, 300.0)):
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.ENEMY)


def _ctl(*, active=True, enabled=True, aimer=None, mouse=None):
    cfg = AimConfig(enabled=enabled)
    sel = TargetSelector(fov_px=400.0, retain_fov_px=500.0, dwell_ms=0.0,
                         switch_margin=0.0, clock=lambda: 0)
    mouse = mouse or NullMouseDriver()
    mouse.connect()
    aimer = aimer or FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0)
    ac = AimController(cfg, selector=sel, imm_manager=IMMManager(), aimer=aimer,
                       mouse=mouse, is_aim_active=lambda: active, roi_size=384)
    return ac, mouse


def test_inactive_commands_no_movement():
    ac, mouse = _ctl(active=False)
    for i in range(5):
        ac.update(Tracks((_enemy(),)), i * 8_000_000)
    assert mouse.moves == []
    assert ac.target_id is None


def test_disabled_config_commands_no_movement():
    ac, mouse = _ctl(active=True, enabled=False)
    ac.update(Tracks((_enemy(),)), 0)
    assert mouse.moves == []


def test_active_enemy_to_the_right_commands_positive_dx():
    ac, mouse = _ctl(active=True)
    for i in range(5):
        ac.update(Tracks((_enemy(),)), i * 8_000_000)
    tdx = sum(m[0] for m in mouse.moves)
    assert tdx > 0
    assert ac.target_id == 1


def test_teammate_never_targeted():
    ac, mouse = _ctl(active=True)
    tm = Track(track_id=1, xyxy=(250.0, 180.0, 290.0, 300.0), confidence=0.9,
               class_id=0, team=Team.TEAMMATE)
    ac.update(Tracks((tm,)), 0)
    assert ac.target_id is None
    assert mouse.moves == []


def test_unknown_never_targeted():
    ac, mouse = _ctl(active=True)
    unk = Track(track_id=1, xyxy=(250.0, 180.0, 290.0, 300.0), confidence=0.9,
                class_id=0, team=Team.UNKNOWN)
    ac.update(Tracks((unk,)), 0)
    assert ac.target_id is None
    assert mouse.moves == []


def test_target_switch_resets_aimer():
    class _Rec(Aimer):
        def __init__(self):
            self.resets = 0
        def step(self, crosshair, target_point, dt):
            return (1.0, 0.0)
        def reset(self):
            self.resets += 1
    rec = _Rec()
    ac, _ = _ctl(active=True, aimer=rec)
    ac.update(Tracks((_enemy(1),)), 0)
    after_first = rec.resets
    ac.update(Tracks((_enemy(2),)), 8_000_000)   # id 1 gone, id 2 appears
    assert ac.target_id == 2
    assert rec.resets > after_first
