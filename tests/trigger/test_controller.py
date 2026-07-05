from ragnarok.config.schema import AimConfig
from ragnarok.core.types import Track, Team
from ragnarok.trigger.controller import TriggerController


class _Mouse:
    def __init__(self):
        self.moves = []
        self.buttons = []
    def move_relative(self, dx, dy):
        self.moves.append((dx, dy))
    def set_button(self, button, down):
        self.buttons.append((button, down))


class _Recoil:
    def __init__(self):
        self.fires = 0
        self.released = 0
    def on_fire(self):
        self.fires += 1
        return (0.0, 4.0)            # 4px vertical counter-move
    def release(self):
        self.released += 1


def _enemy_at_center(roi=100):
    c = roi / 2.0
    return Track(track_id=7, xyxy=(c - 10, c - 10, c + 10, c + 10), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _cfg():
    # hfov/screen_width/sensitivity give a known px->counts factor k
    return AimConfig(hfov_deg=90.0, screen_width_px=900, sensitivity=1.0)


def test_fires_when_enemy_under_crosshair_and_active():
    mouse = _Mouse()
    fired = {"n": 0}
    class _Trig:
        def update(self, **kw):
            fired["n"] += 1
            assert kw["active"] is True and kw["enemy_confirmed"] is True
            return True
        def release(self):
            pass
    tc = TriggerController(_cfg(), trigger=_Trig(), trigger_active=lambda: True,
                           mouse=mouse, roi_size=100, recoil=_Recoil())
    tc.update((_enemy_at_center(),), t_capture_ns=0)
    assert fired["n"] == 1
    assert tc.target_id == 7


def test_recoil_counter_move_applied_on_fire():
    mouse = _Mouse()
    class _Trig:
        def update(self, **kw):
            return True
        def release(self):
            pass
    # k = (90/900)/1.0 = 0.1 deg/px; recoil (0,4)px -> (0, 0.4) counts
    tc = TriggerController(_cfg(), trigger=_Trig(), trigger_active=lambda: True,
                           mouse=mouse, roi_size=100, recoil=_Recoil())
    tc.update((_enemy_at_center(),), t_capture_ns=0)
    assert mouse.moves == [(0.0, 0.4)]


def test_no_enemy_under_crosshair_releases():
    mouse = _Mouse()
    recoil = _Recoil()
    rel = {"n": 0}
    class _Trig:
        def update(self, **kw):
            raise AssertionError("should not fire without an enemy under the crosshair")
        def release(self):
            rel["n"] += 1
    off = Track(track_id=1, xyxy=(0.0, 0.0, 5.0, 5.0), confidence=0.9,
                class_id=0, team=Team.ENEMY)              # far from center(50,50)
    tc = TriggerController(_cfg(), trigger=_Trig(), trigger_active=lambda: True,
                           mouse=mouse, roi_size=100, recoil=recoil)
    tc.update((off,), t_capture_ns=0)
    assert rel["n"] == 1 and recoil.released == 1
    assert tc.target_id is None
    assert mouse.moves == []


def test_teammate_under_crosshair_is_ignored():
    mouse = _Mouse()
    class _Trig:
        def update(self, **kw):
            raise AssertionError("should not fire on a teammate")
        def release(self):
            pass
    c = 50.0
    mate = Track(track_id=2, xyxy=(c - 10, c - 10, c + 10, c + 10), confidence=0.9,
                 class_id=0, team=Team.TEAMMATE)
    tc = TriggerController(_cfg(), trigger=_Trig(), trigger_active=lambda: True,
                           mouse=mouse, roi_size=100)
    tc.update((mate,), t_capture_ns=0)
    assert tc.target_id is None
