"""Phase 9P — the trigger fires independent of the aim key (spec §3.7)."""
from __future__ import annotations

from ragnarok.aim.controller import AimController
from ragnarok.aim.imm import IMMManager
from ragnarok.config.schema import AimConfig
from ragnarok.core.types import Track, Tracks, Team


class _Mouse:
    def __init__(self):
        self.moves = []
        self.buttons = []

    def move_relative(self, dx, dy):
        self.moves.append((dx, dy))

    def set_button(self, b, d):
        self.buttons.append((b, d))


class _Sel:
    def select(self, tracks, cx, cy):
        return None            # no aim lock — isolate the trigger path

    def reset(self):
        pass


class _Trig:
    def __init__(self):
        self.calls = []
        self.is_firing = False

    def update(self, **kw):
        self.calls.append(kw)
        return True

    def release(self):
        pass


def _enemy_center(roi=100):
    c = roi / 2.0
    return Track(track_id=3, xyxy=(c - 10, c - 10, c + 10, c + 10), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _cfg(enabled):
    return AimConfig(enabled=enabled, hfov_deg=90.0, screen_width_px=900, sensitivity=1.0)


def test_trigger_fires_with_aim_disabled():
    trig = _Trig()
    c = AimController(_cfg(enabled=False), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: False,
                      roi_size=100, trigger=trig, trigger_active=lambda: True)
    c.update(Tracks(items=(_enemy_center(),)), t_capture_ns=0)
    assert len(trig.calls) == 1 and trig.calls[0]["active"] is True
    assert c.fire_target_id == 3


def test_trigger_fires_with_aim_enabled_but_key_up():
    trig = _Trig()
    c = AimController(_cfg(enabled=True), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: False,   # aim key UP
                      roi_size=100, trigger=trig, trigger_active=lambda: True)
    c.update(Tracks(items=(_enemy_center(),)), t_capture_ns=0)
    assert len(trig.calls) == 1               # trigger still fired despite aim key up


def test_no_enemy_under_crosshair_clears_fire_target():
    trig = _Trig()
    off = Track(track_id=9, xyxy=(0.0, 0.0, 5.0, 5.0), confidence=0.9, class_id=0, team=Team.ENEMY)
    c = AimController(_cfg(enabled=False), selector=_Sel(), imm_manager=IMMManager(),
                      aimer=None, mouse=_Mouse(), is_aim_active=lambda: False,
                      roi_size=100, trigger=trig, trigger_active=lambda: True)
    c.update(Tracks(items=(off,)), t_capture_ns=0)     # enemy far from centre (50,50)
    assert c.fire_target_id is None
    assert trig.calls == []                              # trigger not evaluated without a target
