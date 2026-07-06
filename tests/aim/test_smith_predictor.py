"""Dead-time compensation: the controller advances the crosshair fed to the aimer
by the counts already commanded (in-flight) but not yet visible in the detection."""
from ragnarok.core.types import Track, Tracks, Team
from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import Aimer
from ragnarok.aim.mouse import NullMouseDriver
from ragnarok.tracking.egomotion import CommandedMotionBuffer


class _RecAimer(Aimer):
    def __init__(self):
        self.calls = []

    def step(self, crosshair, target, dt, target_vel=(0.0, 0.0)):
        self.calls.append((crosshair, target))
        return (0.0, 0.0)             # no output: isolate the crosshair-advance logic

    def reset(self):
        pass


def _enemy():
    return Track(track_id=1, xyxy=(250.0, 180.0, 290.0, 300.0), confidence=0.9,
                 class_id=0, team=Team.ENEMY)


def _controller(cfg, buf):
    sel = TargetSelector(fov_px=400.0, retain_fov_px=500.0, dwell_ms=0.0,
                         switch_margin=0.0, clock=lambda: 0)
    mouse = NullMouseDriver()
    mouse.connect()
    rec = _RecAimer()
    ac = AimController(cfg, selector=sel, imm_manager=IMMManager(), aimer=rec, mouse=mouse,
                       is_aim_active=lambda: True, roi_size=384,
                       clock=lambda: 100_000_000, commanded_buffer=buf)   # now = 100ms
    return ac, rec


def _inv_k(cfg):
    return cfg.sensitivity / (cfg.hfov_deg / cfg.screen_width_px)          # counts -> px


def test_crosshair_advanced_by_inflight_counts():
    cfg = AimConfig(enabled=True, deadtime_ms=50.0, adaptive_lead=False, lead_ms=0.0)
    buf = CommandedMotionBuffer()
    buf.push(80_000_000, 100.0, 0.0)          # 100 counts right, inside [50ms,100ms] window
    ac, rec = _controller(cfg, buf)
    ac.update(Tracks((_enemy(),)), 0)
    ch, tgt = rec.calls[-1]
    assert abs(ch[0] - (192.0 + 100.0 * _inv_k(cfg))) < 1e-6    # advanced by in-flight px
    assert ch[0] <= tgt[0]                                       # but not past the target


def test_overestimated_deadtime_clamps_at_target():
    cfg = AimConfig(enabled=True, deadtime_ms=50.0, adaptive_lead=False, lead_ms=0.0)
    buf = CommandedMotionBuffer()
    buf.push(80_000_000, 100000.0, 0.0)        # absurd in-flight -> would overshoot
    ac, rec = _controller(cfg, buf)
    ac.update(Tracks((_enemy(),)), 0)
    ch, tgt = rec.calls[-1]
    assert abs(ch[0] - tgt[0]) < 1e-6          # clamped to target, never reverses past it


def test_deadtime_zero_leaves_crosshair_at_center():
    cfg = AimConfig(enabled=True, deadtime_ms=0.0, adaptive_lead=False, lead_ms=0.0)
    buf = CommandedMotionBuffer()
    buf.push(80_000_000, 100.0, 0.0)
    ac, rec = _controller(cfg, buf)
    ac.update(Tracks((_enemy(),)), 0)
    ch, _ = rec.calls[-1]
    assert ch[0] == 192.0                       # compensation off -> ROI center
