import pytest
from pydantic import ValidationError
from ragnarok.config.schema import (
    AimConfig, MotionConfig, RecoilConfig, TriggerConfig, AppConfig,
)


def test_aim_new_defaults():
    a = AimConfig()
    assert a.kff == 0.0
    assert a.vel_clamp_px_s == 4000.0
    assert a.vel_smooth_alpha == 0.5
    assert a.hybrid_flick_dist_px == 20.0
    assert a.adaptive_lead is True
    assert a.lead_alpha == 0.1


def test_aimer_accepts_hybrid_and_predictive():
    assert AimConfig(aimer="hybrid").aimer == "hybrid"
    assert AimConfig(aimer="predictive").aimer == "predictive"


def test_aimer_rejects_unknown():
    with pytest.raises(ValidationError):
        AimConfig(aimer="magic")


def test_motion_defaults():
    m = MotionConfig()
    assert m.shaper == "none"
    assert m.gravity == 9.0 and m.wind == 3.0
    assert m.max_step == 15.0 and m.target_area == 10.0


def test_recoil_defaults_and_pattern():
    r = RecoilConfig(pattern=((0.0, 0.0), (0.0, 10.0)))
    assert r.enabled is False and r.scale == 1.0
    assert r.pattern == ((0.0, 0.0), (0.0, 10.0))


def test_trigger_defaults():
    t = TriggerConfig()
    assert t.enabled is False
    assert t.trigger_key == "VK_LBUTTON"
    assert t.activation_delay_ms == 80.0
    assert t.require_line_clear is True
    assert t.button == "left"


def test_appconfig_nests_phase4_sections():
    app = AppConfig()
    assert isinstance(app.motion, MotionConfig)
    assert isinstance(app.recoil, RecoilConfig)
    assert isinstance(app.trigger, TriggerConfig)


def test_backward_compat_without_phase4_sections():
    app = AppConfig(detection={"backend": "rfdetr_torch", "model": "nano"})
    assert app.motion.shaper == "none"
    assert app.trigger.enabled is False


def test_frozen():
    with pytest.raises(Exception):
        MotionConfig().shaper = "windmouse"
