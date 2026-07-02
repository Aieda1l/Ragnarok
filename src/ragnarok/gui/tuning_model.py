"""Pure binding layer for the live tuning panels (spec §13).

ZERO Qt: describes each tunable field (``FieldSpec``) and applies an edit by
building a NEW, RE-VALIDATED frozen ``AppConfig``. pydantic v2's
``model_copy(update=...)`` skips validation, so ``set_field`` reconstructs the
edited sub-model through its class — an out-of-range value raises
``ValidationError`` and never reaches the ``ConfigHandle``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle


@dataclass(frozen=True)
class FieldSpec:
    path: str                     # "section.field"
    label: str
    kind: str                     # "float" | "int" | "bool" | "choice"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()


# The "Aim" tab. Ranges mirror config.schema.AimConfig Field() constraints.
AIM_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("aim.enabled", "Aim enabled", "bool"),
    FieldSpec("aim.aimer", "Aimer", "choice",
              choices=("flick", "feedback", "hybrid", "predictive")),
    FieldSpec("aim.controller_mode", "PID mode", "choice", choices=("p", "pi", "pid")),
    FieldSpec("aim.kp", "Kp", "float", 0.01, 2.0, 0.01),
    FieldSpec("aim.ki", "Ki", "float", 0.0, 5.0, 0.01),
    FieldSpec("aim.kd", "Kd", "float", 0.0, 5.0, 0.01),
    FieldSpec("aim.kff", "Kff (feed-fwd)", "float", 0.0, 4.0, 0.05),
    FieldSpec("aim.max_step_px", "Max step (px)", "float", 1.0, 300.0, 1.0),
    FieldSpec("aim.ema_alpha", "EMA alpha", "float", 0.01, 1.0, 0.01),
    FieldSpec("aim.aim_fov_deg", "FOV acquire (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.retain_fov_deg", "FOV retain (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.dwell_ms", "Dwell (ms)", "float", 0.0, 2000.0, 10.0),
    FieldSpec("aim.switch_margin", "Switch margin", "float", 0.0, 0.99, 0.01),
    FieldSpec("aim.sensitivity", "Sensitivity (deg/count)", "float", 0.001, 1.0, 0.001),
    FieldSpec("aim.lead_ms", "Lead (ms)", "float", 0.0, 500.0, 5.0),
    FieldSpec("aim.head_frac", "Head fraction", "float", 0.0, 1.0, 0.01),
    FieldSpec("aim.aim_point", "Aim point", "choice", choices=("head", "body")),
    FieldSpec("aim.adaptive_lead", "Adaptive lead", "bool"),
)


def _split(path: str) -> tuple[str, str]:
    section, field = path.split(".", 1)
    return section, field


def get_field(cfg: AppConfig, path: str):
    section, field = _split(path)
    return getattr(getattr(cfg, section), field)


def set_field(cfg: AppConfig, path: str, value) -> AppConfig:
    """Return a NEW frozen AppConfig with ``path`` set to ``value``.

    Re-validates by reconstructing the sub-model through its class; an invalid
    value raises ``pydantic.ValidationError``.
    """
    section, field = _split(path)
    sub = getattr(cfg, section)
    new_sub = sub.__class__(**{**sub.model_dump(), field: value})   # validates
    return cfg.model_copy(update={section: new_sub})


def apply_field(handle: ConfigHandle, path: str, value) -> AppConfig:
    new_cfg = set_field(handle.current, path, value)
    handle.swap(new_cfg)
    return new_cfg
