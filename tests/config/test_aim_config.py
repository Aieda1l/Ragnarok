"""Tests for AimConfig and its nesting in AppConfig (Task 1 — Phase 3 aim core)."""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from ragnarok.config.schema import AimConfig, AppConfig


class TestAimConfigDefaults:
    def test_enabled_default_false(self):
        assert AimConfig().enabled is False

    def test_aim_key_default(self):
        assert AimConfig().aim_key == "VK_XBUTTON2"

    def test_toggle_default_true(self):
        assert AimConfig().toggle is True

    def test_hfov_deg_default(self):
        assert AimConfig().hfov_deg == 90.0

    def test_screen_width_px_default(self):
        assert AimConfig().screen_width_px == 1920

    def test_aim_fov_deg_default(self):
        assert AimConfig().aim_fov_deg == 5.0

    def test_retain_fov_deg_default(self):
        assert AimConfig().retain_fov_deg == 8.0

    def test_dwell_ms_default(self):
        assert AimConfig().dwell_ms == 100.0

    def test_switch_margin_default(self):
        assert AimConfig().switch_margin == 0.20

    def test_aimer_default(self):
        assert AimConfig().aimer == "feedback"

    def test_kp_default(self):
        assert AimConfig().kp == 0.35

    def test_max_step_px_default(self):
        assert AimConfig().max_step_px == 60.0

    def test_flick_speed_px_s_default(self):
        assert AimConfig().flick_speed_px_s == 4000.0

    def test_ema_alpha_default(self):
        assert AimConfig().ema_alpha == 0.5

    def test_aim_point_default(self):
        assert AimConfig().aim_point == "head"

    def test_head_frac_default(self):
        assert AimConfig().head_frac == 0.15

    def test_sensitivity_default(self):
        assert AimConfig().sensitivity == 0.022

    def test_lead_ms_default(self):
        assert AimConfig().lead_ms == 0.0


class TestAimConfigFrozen:
    def test_cannot_set_enabled(self):
        cfg = AimConfig()
        with pytest.raises(Exception):
            cfg.enabled = True  # type: ignore[misc]

    def test_cannot_set_hfov_deg(self):
        cfg = AimConfig()
        with pytest.raises(Exception):
            cfg.hfov_deg = 120.0  # type: ignore[misc]

    def test_cannot_set_aimer(self):
        cfg = AimConfig()
        with pytest.raises(Exception):
            cfg.aimer = "flick"  # type: ignore[misc]


class TestAimConfigValidation:
    def test_bad_aimer_rejected(self):
        with pytest.raises(ValidationError):
            AimConfig(aimer="turbo")  # type: ignore[call-arg]

    def test_bad_aim_point_rejected(self):
        with pytest.raises(ValidationError):
            AimConfig(aim_point="torso")  # type: ignore[call-arg]

    def test_hfov_deg_must_be_positive(self):
        with pytest.raises(ValidationError):
            AimConfig(hfov_deg=0.0)

    def test_hfov_deg_max_180(self):
        with pytest.raises(ValidationError):
            AimConfig(hfov_deg=181.0)

    def test_screen_width_px_min(self):
        with pytest.raises(ValidationError):
            AimConfig(screen_width_px=319)

    def test_switch_margin_must_be_lt_1(self):
        with pytest.raises(ValidationError):
            AimConfig(switch_margin=1.0)

    def test_sensitivity_must_be_positive(self):
        with pytest.raises(ValidationError):
            AimConfig(sensitivity=0.0)

    def test_flick_aimer_valid(self):
        cfg = AimConfig(aimer="flick")
        assert cfg.aimer == "flick"

    def test_feedback_aimer_valid(self):
        cfg = AimConfig(aimer="feedback")
        assert cfg.aimer == "feedback"


class TestAppConfigNesting:
    def test_appconfig_has_aim_field(self):
        app = AppConfig()
        assert hasattr(app, "aim")

    def test_appconfig_aim_is_aim_config(self):
        app = AppConfig()
        assert isinstance(app.aim, AimConfig)

    def test_appconfig_aim_aimer_default(self):
        assert AppConfig().aim.aimer == "feedback"

    def test_appconfig_aim_enabled_default_false(self):
        assert AppConfig().aim.enabled is False

    def test_existing_appconfig_still_valid(self):
        """Backward compat: AppConfig() with only capture/detection still works."""
        app = AppConfig(
            capture={"backend": "mss", "roi_size": 384, "target_fps": 60, "monitor_index": 0},
            detection={"backend": "rfdetr_torch", "model": "nano", "confidence": 0.6},
        )
        assert app.capture.backend == "mss"
        assert app.detection.model == "nano"
        assert app.aim.aimer == "feedback"

    def test_appconfig_aim_can_be_customised(self):
        app = AppConfig(aim={"enabled": True, "aimer": "flick", "kp": 0.5})
        assert app.aim.enabled is True
        assert app.aim.aimer == "flick"
        assert app.aim.kp == 0.5
