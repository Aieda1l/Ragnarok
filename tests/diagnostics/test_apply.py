"""Tests for apply_seeds (explicit, immutable config update)."""
from __future__ import annotations
from ragnarok.config.schema import AppConfig
from ragnarok.diagnostics.numeric_tune import PidSeeds
from ragnarok.diagnostics.apply import apply_seeds


def test_apply_seeds_returns_new_config_with_gains():
    cfg = AppConfig()
    seeds = PidSeeds(kp=0.5, ki=0.2, kd=0.05)
    out = apply_seeds(cfg, seeds, controller_mode="pid")
    assert out.aim.kp == 0.5 and out.aim.ki == 0.2 and out.aim.kd == 0.05
    assert out.aim.controller_mode == "pid"


def test_original_config_is_unchanged():
    cfg = AppConfig()
    apply_seeds(cfg, PidSeeds(kp=9.9, ki=9.9, kd=9.9))
    assert cfg.aim.kp == 0.35 and cfg.aim.ki == 0.0   # original defaults intact
