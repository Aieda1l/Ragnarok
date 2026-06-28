"""Tests for ClassificationConfig and its nesting in AppConfig (live-wiring follow-up)."""
from __future__ import annotations
import pytest
from pydantic import ValidationError

from ragnarok.config.schema import ClassificationConfig, AppConfig


class TestClassificationConfigDefaults:
    def test_enabled_default_true(self):
        assert ClassificationConfig().enabled is True

    def test_palette_default(self):
        assert ClassificationConfig().palette == "default"

    def test_enemy_color_default(self):
        assert ClassificationConfig().enemy_color == "red"

    def test_frac_threshold_default(self):
        assert ClassificationConfig().frac_threshold == 0.18

    def test_thickness_default(self):
        assert ClassificationConfig().thickness == 4

    def test_vote_window_default(self):
        assert ClassificationConfig().vote_window == 5

    def test_vote_min_default(self):
        assert ClassificationConfig().vote_min == 3


class TestClassificationConfigValidation:
    def test_bad_palette_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationConfig(palette="rainbow")  # type: ignore[arg-type]

    def test_wong_palette_valid(self):
        assert ClassificationConfig(palette="wong").palette == "wong"

    def test_frac_threshold_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationConfig(frac_threshold=1.5)

    def test_thickness_must_be_positive(self):
        with pytest.raises(ValidationError):
            ClassificationConfig(thickness=0)

    def test_vote_min_must_be_positive(self):
        with pytest.raises(ValidationError):
            ClassificationConfig(vote_min=0)

    def test_frozen(self):
        cfg = ClassificationConfig()
        with pytest.raises(Exception):
            cfg.enabled = False  # type: ignore[misc]


class TestAppConfigNesting:
    def test_appconfig_has_classification_field(self):
        assert isinstance(AppConfig().classification, ClassificationConfig)

    def test_appconfig_classification_default_enabled(self):
        assert AppConfig().classification.enabled is True

    def test_can_be_customised(self):
        app = AppConfig(classification={"enabled": False, "palette": "wong",
                                        "enemy_color": "orange"})
        assert app.classification.enabled is False
        assert app.classification.palette == "wong"
        assert app.classification.enemy_color == "orange"

    def test_backward_compat_without_classification(self):
        app = AppConfig(detection={"backend": "rfdetr_torch", "model": "nano"})
        assert app.classification.enabled is True
