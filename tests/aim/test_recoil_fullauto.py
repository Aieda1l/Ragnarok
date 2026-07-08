"""Full-auto recoil: the spray pattern advances per-shot at fire_rate_rps while
the trigger is held; semi-auto (rps=0) advances only once per new press."""
from ragnarok.core.types import Track, Tracks, Team
from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.aim.mouse import NullMouseDriver
from ragnarok.trigger.bot import TriggerBot
from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator


class _FakeTrigger:
    def __init__(self, fired_seq):
        self._seq = list(fired_seq)
        self.is_firing = True

    def update(self, **kw):
        return self._seq.pop(0) if self._seq else False

    def release(self):
        pass


class _FakeRecoil:
    def __init__(self, rps):
        self.fire_rate_rps = rps
        self.fires = 0
        self.releases = 0

    def on_fire(self):
        self.fires += 1
        return (0.0, 0.0)

    def release(self):
        self.releases += 1


def _enemy():
    # box covers the ROI-centre crosshair (384/2 = 192,192) so the crosshair-
    # containment trigger fires (Phase 9P: trigger fires on crosshair-over-enemy).
    return Track(track_id=1, xyxy=(150.0, 150.0, 250.0, 300.0), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _off_enemy():
    # enemy away from the crosshair -> no target under it -> spray releases.
    return Track(track_id=1, xyxy=(300.0, 300.0, 340.0, 360.0), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _drive(rps, ticks, dt_ns):
    clock = {"t": 0}
    cfg = AimConfig(enabled=True, adaptive_lead=False, lead_ms=0.0)
    sel = TargetSelector(fov_px=400.0, retain_fov_px=500.0, dwell_ms=0.0,
                         switch_margin=0.0, clock=lambda: 0)
    mouse = NullMouseDriver()
    mouse.connect()
    trig = _FakeTrigger([True] + [False] * (ticks + 5))
    rec = _FakeRecoil(rps)
    ac = AimController(cfg, selector=sel, imm_manager=IMMManager(),
                       aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
                       mouse=mouse, is_aim_active=lambda: True, roi_size=384,
                       clock=lambda: clock["t"], trigger=trig, trigger_active=lambda: True,
                       recoil=rec)
    for _ in range(ticks):
        ac.update(Tracks((_enemy(),)), clock["t"])
        clock["t"] += dt_ns
    # Final tick with no enemy under the crosshair -> the spray releases (Phase 9P:
    # recoil release is driven by the trigger path, not aim-target acquisition).
    ac.update(Tracks((_off_enemy(),)), clock["t"])
    return rec


def test_full_auto_advances_at_fire_rate():
    # rps=10 -> shot every 100ms; ticks at 10ms over t=0..1000ms -> shots at 0,100,...,1000 = 11
    rec = _drive(rps=10.0, ticks=101, dt_ns=10_000_000)
    assert rec.fires == 11
    assert rec.releases >= 1                     # burst restarts from shot 0


def test_semi_auto_advances_once_per_press():
    rec = _drive(rps=0.0, ticks=101, dt_ns=10_000_000)
    assert rec.fires == 1                        # only the initial press; no held advance


def test_trigger_bot_is_firing_reflects_press():
    events = []
    mouse = NullMouseDriver()
    mouse.connect()
    bot = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    assert bot.is_firing is False
    trk = _enemy()                                  # box (150,150,250,300)
    bot.update(track=trk, crosshair=(200.0, 240.0), occluded=False,
               enemy_confirmed=True, line_clear=True, active=True)
    assert bot.is_firing is True
    bot.release()
    assert bot.is_firing is False


def test_compensator_stores_fire_rate():
    c = RecoilCompensator(RecoilPattern(points=((0.0, 2.0),)), scale=1.0, fire_rate_rps=8.0)
    assert c.fire_rate_rps == 8.0
