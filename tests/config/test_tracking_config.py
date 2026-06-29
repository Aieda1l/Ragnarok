"""Tests for TrackingConfig and its nesting in AppConfig (live-wiring follow-up)."""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from ragnarok.config.schema import TrackingConfig, AppConfig


class TestTrackingConfigDefaults:
    def test_backend_default_botsort(self):
        assert TrackingConfig().backend == "botsort"

    def test_track_high_thresh_default(self):
        assert TrackingConfig().track_high_thresh == 0.6

    def test_track_low_thresh_default(self):
        assert TrackingConfig().track_low_thresh == 0.1

    def test_new_track_thresh_default(self):
        assert TrackingConfig().new_track_thresh == 0.7

    def test_track_buffer_default(self):
        assert TrackingConfig().track_buffer == 30

    def test_match_thresh_default(self):
        assert TrackingConfig().match_thresh == 0.8

    def test_proximity_thresh_default(self):
        assert TrackingConfig().proximity_thresh == 0.5


class TestTrackingConfigValidation:
    def test_bad_backend_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(backend="sort")  # type: ignore[arg-type]

    def test_identity_backend_valid(self):
        assert TrackingConfig(backend="identity").backend == "identity"

    def test_thresh_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(track_high_thresh=1.5)

    def test_track_buffer_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrackingConfig(track_buffer=0)

    def test_frozen(self):
        cfg = TrackingConfig()
        with pytest.raises(Exception):
            cfg.backend = "identity"  # type: ignore[misc]


class TestAppConfigNesting:
    def test_appconfig_has_tracking_field(self):
        assert isinstance(AppConfig().tracking, TrackingConfig)

    def test_appconfig_tracking_default_botsort(self):
        assert AppConfig().tracking.backend == "botsort"

    def test_can_be_customised(self):
        app = AppConfig(tracking={"backend": "identity", "track_buffer": 60})
        assert app.tracking.backend == "identity"
        assert app.tracking.track_buffer == 60

    def test_backward_compat_without_tracking(self):
        app = AppConfig(
            capture={"backend": "mss", "roi_size": 384, "target_fps": 60, "monitor_index": 0},
        )
        assert app.tracking.backend == "botsort"


def test_gmc_defaults_off():
    assert TrackingConfig().gmc == "off"
    assert TrackingConfig().deg_per_count == 0.0
    assert TrackingConfig().tau_render_s == 0.0


def test_gmc_feedforward_and_signed_deg_per_count():
    t = TrackingConfig(gmc="feedforward", deg_per_count=-0.022, tau_render_s=0.012)
    assert t.gmc == "feedforward"
    assert t.deg_per_count == -0.022          # signed: negative is valid
    assert t.tau_render_s == 0.012


def test_gmc_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        TrackingConfig(gmc="optical")  # type: ignore[arg-type]


def test_tau_render_bounds():
    with pytest.raises(ValidationError):
        TrackingConfig(tau_render_s=-0.001)
