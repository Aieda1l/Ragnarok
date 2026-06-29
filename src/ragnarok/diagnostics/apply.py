"""Apply auto-tune SEEDS into a new frozen AppConfig (spec §11: seeds, not final).

Explicit only — never auto-writes config or touches disk. The caller wires the
returned AppConfig into ConfigHandle.swap when (and if) the user accepts it.
"""
from __future__ import annotations

from ragnarok.config.schema import AppConfig
from ragnarok.diagnostics.numeric_tune import PidSeeds


def apply_seeds(cfg: AppConfig, seeds: PidSeeds, *, controller_mode: str = "pid") -> AppConfig:
    new_aim = cfg.aim.model_copy(update={
        "kp": seeds.kp, "ki": seeds.ki, "kd": seeds.kd,
        "controller_mode": controller_mode,
    })
    return cfg.model_copy(update={"aim": new_aim})
