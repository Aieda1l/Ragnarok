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
    kind: str                     # "float" | "int" | "bool" | "choice" | "text"
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
    FieldSpec("aim.creep_px", "Creep zone (px)", "float", 0.0, 100.0, 1.0),
    FieldSpec("aim.commit", "Commit fraction", "float", 0.05, 1.0, 0.05),
    FieldSpec("aim.settle_px", "Settle deadzone (px)", "float", 0.0, 20.0, 0.5),
    FieldSpec("aim.deadtime_ms", "Latency comp (ms)", "float", 0.0, 500.0, 5.0),
    FieldSpec("aim.pinhole", "Pinhole projection", "bool"),
    FieldSpec("aim.ema_alpha", "EMA alpha", "float", 0.01, 1.0, 0.01),
    FieldSpec("aim.aim_fov_deg", "FOV acquire (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.retain_fov_deg", "FOV retain (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.dwell_ms", "Dwell (ms)", "float", 0.0, 2000.0, 10.0),
    FieldSpec("aim.switch_margin", "Switch margin", "float", 0.0, 0.99, 0.01),
    FieldSpec("aim.sensitivity", "Sensitivity (deg/count)", "float", 0.001, 1.0, 0.001),
    FieldSpec("aim.lead_ms", "Lead (ms)", "float", 0.0, 500.0, 5.0),
    FieldSpec("aim.head_frac", "Head fraction", "float", 0.0, 1.0, 0.01),
    FieldSpec("aim.aim_point", "Aim point", "choice",
              choices=("head", "body", "detected_head")),
    FieldSpec("aim.head_class_id", "Head class id", "int", 0, 90, 1),
    FieldSpec("aim.adaptive_lead", "Adaptive lead", "bool"),
)


# Ranges mirror config.schema Field() constraints. Detection is box-only (a live
# detector swap reloads the torch/TRT model) and has no tab here.
TRACKING_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("tracking.backend", "Tracker", "choice", choices=("botsort", "identity")),
    FieldSpec("tracking.track_high_thresh", "Track high thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.track_low_thresh", "Track low thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.new_track_thresh", "New-track thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.track_buffer", "Track buffer", "int", 1, 600, 1),
    FieldSpec("tracking.match_thresh", "Match thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.proximity_thresh", "Proximity thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.gmc", "GMC", "choice", choices=("off", "feedforward")),
    # deg_per_count is schema-unbounded and signed; the sane default range below
    # is auto-expanded by TuningPanel._fit_range for any larger loaded value.
    FieldSpec("tracking.deg_per_count", "deg/count (signed)", "float", -1.0, 1.0, 0.001),
    FieldSpec("tracking.tau_render_s", "τ_render (s)", "float", 0.0, 0.1, 0.001),
)

# enemy_color is exposed with the "default" palette keys. The colorblind "wong"
# palette (and its color keys) stays TOML-only: switching palette without a
# matching color key would break resolve_enemy_profile, and that combo needs the
# eyedropper wizard's palette-aware choices rather than a fixed tuple.
CLASSIFICATION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("classification.enabled", "Friend/Foe enabled", "bool"),
    FieldSpec("classification.enemy_color", "Enemy color", "choice",
              choices=("red", "purple", "yellow")),
    FieldSpec("classification.frac_threshold", "Outline frac thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("classification.thickness", "Ring thickness (px)", "int", 1, 64, 1),
    FieldSpec("classification.vote_window", "Vote window", "int", 1, 120, 1),
    FieldSpec("classification.vote_min", "Vote min", "int", 1, 120, 1),
)

TRIGGER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("trigger.enabled", "Trigger bot enabled", "bool"),
    FieldSpec("trigger.activation_delay_ms", "Activation delay (ms)", "float", 0.0, 2000.0, 5.0),
    FieldSpec("trigger.max_occlusion_frames", "Occlusion tolerance (frames)", "int", 0, 30, 1),
    FieldSpec("trigger.require_line_clear", "Require line clear", "bool"),
    FieldSpec("trigger.button", "Fire button", "choice", choices=("left", "right", "middle")),
)

# recoil.scale / motion.gravity / motion.wind are schema-unbounded above (ge=0.0,
# no le); the sane default caps below are auto-expanded by TuningPanel._fit_range
# for any larger loaded value, so nothing is ever silently clamped/downgraded.
RECOIL_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("recoil.enabled", "Recoil comp enabled", "bool"),
    FieldSpec("recoil.scale", "Recoil scale", "float", 0.0, 5.0, 0.1),
)

MOTION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("motion.shaper", "Motion shaper", "choice", choices=("none", "windmouse")),
    FieldSpec("motion.gravity", "WindMouse gravity", "float", 0.0, 50.0, 0.5),
    FieldSpec("motion.wind", "WindMouse wind", "float", 0.0, 20.0, 0.5),
    FieldSpec("motion.max_step", "WindMouse max step", "float", 1.0, 100.0, 1.0),
    FieldSpec("motion.target_area", "WindMouse target area", "float", 1.0, 100.0, 1.0),
)

# Input: which mouse driver + the Arduino transport settings. The hardware
# test-move is box-only; here we only bind config.
INPUT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("input.mouse_driver", "Mouse driver", "choice",
              choices=("sendinput", "arduino")),
    FieldSpec("input.compensate_ballistics", "Compensate ptr ballistics", "bool"),
    FieldSpec("arduino.transport", "Arduino transport", "choice",
              choices=("serial", "udp")),
    FieldSpec("arduino.port", "Serial port (COM/tty)", "text"),
    FieldSpec("arduino.baud", "Baud", "int", 1200, 2000000, 100),
    FieldSpec("arduino.host", "UDP host", "text"),
    FieldSpec("arduino.udp_port", "UDP port", "int", 0, 65535, 1),
)

DETECTION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("detection.confidence", "Confidence threshold", "float", 0.0, 1.0, 0.01),
)

OVERLAY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("overlay.show_confidence", "Show confidence", "bool"),
    FieldSpec("overlay.show_fov", "Show FOV brackets", "bool"),
    FieldSpec("overlay.show_boxes", "Show non-enemy boxes", "bool"),
    FieldSpec("overlay.show_tracking_line", "Show tracking line", "bool"),
    FieldSpec("overlay.diamond_scale", "Diamond size", "float", 0.2, 5.0, 0.1),
    FieldSpec("overlay.scanlines", "CRT scanlines", "bool"),
    FieldSpec("overlay.chroma", "Chromatic aberration", "bool"),
)

KEYBIND_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("aim.aim_key", "Aim toggle key (VK_ or char)", "text"),
    FieldSpec("aim.toggle", "Aim = toggle (off = hold)", "bool"),
    FieldSpec("trigger.trigger_key", "Trigger toggle key (VK_ or char)", "text"),
    FieldSpec("calibration.reset_key", "Calib reset key (VK_)", "text"),
    FieldSpec("calibration.apply_key", "Calib set-sens key (VK_)", "text"),
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
