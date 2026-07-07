import math

from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import NullAimer
from ragnarok.aim.mouse import NullMouseDriver


def _ctl(pinhole):
    cfg = AimConfig(enabled=True, hfov_deg=90.0, screen_width_px=1920, pinhole=pinhole)
    m = NullMouseDriver()
    m.connect()
    sel = TargetSelector(fov_px=1.0, retain_fov_px=1.0, dwell_ms=0.0,
                         switch_margin=0.0, clock=lambda: 0)
    return AimController(cfg, selector=sel, imm_manager=IMMManager(), aimer=NullAimer(),
                         mouse=m, is_aim_active=lambda: True, roi_size=384)


def test_linear_vs_pinhole_deg_per_px():
    assert abs(_ctl(False)._deg_per_px - 90.0 / 1920.0) < 1e-9      # linear default
    focal = (1920 / 2.0) / math.tan(math.radians(45.0))            # = 960
    assert abs(_ctl(True)._deg_per_px - math.degrees(1.0 / focal)) < 1e-9
    assert _ctl(True)._deg_per_px > _ctl(False)._deg_per_px         # pinhole larger at 90°
