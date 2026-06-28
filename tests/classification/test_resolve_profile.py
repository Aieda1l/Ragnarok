"""Tests for resolve_enemy_profile (palette + color -> ColorProfile)."""
from __future__ import annotations
import pytest

from ragnarok.classification.color import (
    resolve_enemy_profile, RED, PURPLE, PALETTES,
)


def test_default_red():
    assert resolve_enemy_profile("default", "red") is RED


def test_default_purple():
    assert resolve_enemy_profile("default", "purple") is PURPLE


def test_wong_orange_resolves():
    prof = resolve_enemy_profile("wong", "orange")
    assert prof.name == "orange"


def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="palette"):
        resolve_enemy_profile("rainbow", "red")


def test_unknown_color_raises_with_choices():
    with pytest.raises(ValueError, match="enemy_color"):
        resolve_enemy_profile("default", "chartreuse")


def test_palettes_table_has_both():
    assert "default" in PALETTES and "wong" in PALETTES
