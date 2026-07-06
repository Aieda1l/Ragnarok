from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["bettercam", "mss"] = "bettercam"
    roi_size: int = Field(default=384, ge=64, le=1280)
    target_fps: int = Field(default=144, ge=1, le=1000)
    monitor_index: int = 0


class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["rfdetr_torch", "rfdetr_trt"] = "rfdetr_torch"
    model: Literal["nano", "small", "medium", "large"] = "small"  # Apache-2.0 variants only
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    optimize_fp16: bool = True   # fuse + FP16 the auto-built model (~order-of-magnitude on Ampere)
    engine_path: str = ""        # path to a built TensorRT .engine (rfdetr_trt backend)
    # precision is consumed by the (box-only) export tooling (build_trt_command /
    # engine_path_for); the runtime detector just loads the prebuilt engine_path.
    precision: Literal["fp16", "int8"] = "fp16"   # FP16 default; INT8 box-only (needs modelopt Q/DQ)


class AimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    aim_key: str = "VK_RBUTTON"
    toggle: bool = False                        # False = hold-to-aim
    hfov_deg: float = Field(default=90.0, gt=0.0, le=180.0)
    screen_width_px: int = Field(default=1920, ge=320, le=7680)
    aim_fov_deg: float = Field(default=5.0, gt=0.0, le=179.0)      # acquire (inner)
    retain_fov_deg: float = Field(default=8.0, gt=0.0, le=179.0)   # keep (outer) > inner
    dwell_ms: float = Field(default=100.0, ge=0.0, le=2000.0)
    switch_margin: float = Field(default=0.20, ge=0.0, lt=1.0)
    aimer: Literal["flick", "feedback", "hybrid", "predictive"] = "feedback"
    kp: float = Field(default=0.35, gt=0.0, le=2.0)
    max_step_px: float = Field(default=60.0, gt=0.0)
    flick_speed_px_s: float = Field(default=4000.0, gt=0.0)
    ema_alpha: float = Field(default=0.5, gt=0.0, le=1.0)
    aim_point: Literal["head", "body", "detected_head"] = "head"
    head_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    head_class_id: int = Field(default=1, ge=0)   # detection class = head (for "detected_head")
    sensitivity: float = Field(default=0.022, gt=0.0)              # deg per mouse count
    lead_ms: float = Field(default=40.0, ge=0.0, le=500.0)
    # --- Phase 4 additions ---
    kff: float = Field(default=0.0, ge=0.0, le=4.0)               # feed-forward velocity gain
    vel_clamp_px_s: float = Field(default=4000.0, gt=0.0)         # v̂ saturation
    vel_smooth_alpha: float = Field(default=0.5, gt=0.0, le=1.0)  # v̂ low-pass
    hybrid_flick_dist_px: float = Field(default=20.0, gt=0.0)     # HybridAimer threshold
    adaptive_lead: bool = True                                    # §6.5 adaptive vs fixed lead_ms
    lead_alpha: float = Field(default=0.1, gt=0.0, le=1.0)        # adaptive-lead EWMA
    # --- Phase 5A PID additions ---
    ki: float = Field(default=0.0, ge=0.0)
    kd: float = Field(default=0.0, ge=0.0)
    controller_mode: Literal["p", "pi", "pid"] = "p"
    integral_clamp: float | None = Field(default=None, gt=0.0)
    cond_integ_thresh_px: float | None = Field(default=None, gt=0.0)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: Literal["botsort", "identity"] = "botsort"
    track_high_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.7, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1, le=600)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    proximity_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    # --- Phase 5B feed-forward GMC ---
    gmc: Literal["off", "feedforward"] = "off"
    deg_per_count: float = 0.0          # SIGNED degrees of yaw/pitch per mouse count (empirical)
    tau_render_s: float = Field(default=0.0, ge=0.0, le=0.1)   # render+display latency window


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = True
    palette: Literal["default", "wong"] = "default"
    enemy_color: str = "red"                  # key within the chosen palette
    frac_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    thickness: int = Field(default=4, ge=1, le=64)
    vote_window: int = Field(default=5, ge=1, le=120)
    vote_min: int = Field(default=3, ge=1, le=120)


class MotionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    shaper: Literal["none", "windmouse"] = "none"
    gravity: float = Field(default=9.0, ge=0.0)
    wind: float = Field(default=3.0, ge=0.0)
    max_step: float = Field(default=15.0, gt=0.0)
    target_area: float = Field(default=10.0, gt=0.0)


class RecoilConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    scale: float = Field(default=1.0, ge=0.0)
    pattern: tuple[tuple[float, float], ...] = ()


class TriggerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    trigger_key: str = "VK_LBUTTON"
    activation_delay_ms: float = Field(default=80.0, ge=0.0, le=2000.0)
    require_line_clear: bool = True
    button: Literal["left", "right", "middle"] = "left"


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    step_px: float = Field(default=200.0, gt=0.0)
    sample_hz: float = Field(default=1000.0, gt=0.0)
    timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    settle_band_frac: float = Field(default=0.02, gt=0.0, lt=1.0)
    rise_lo: float = Field(default=0.1, gt=0.0, lt=1.0)
    rise_hi: float = Field(default=0.9, gt=0.0, lt=1.0)
    dead_frac: float = Field(default=0.05, ge=0.0, lt=1.0)
    reg_max_overshoot_pct: float = Field(default=5.0, ge=0.0)


class TrainingConfig(BaseModel):
    """Training-pipeline config (spec §12).

    NOTE: the Roboflow API key is intentionally NOT a field here — it is read
    from the RAGNAROK_ROBOFLOW_API_KEY environment variable by the Roboflow
    client (Plan 6B), so secrets never land in a committed/example config.
    Paths are relative to the Ragnarok app data dir.
    """
    model_config = ConfigDict(frozen=True)
    frames_dir: str = "captures"
    dataset_dir: str = "datasets"
    engines_dir: str = "engines"
    roboflow_workspace: str = ""
    roboflow_project: str = ""
    roboflow_version: int = Field(default=1, ge=1)
    capture_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    scene_change_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    min_capture_interval_s: float = Field(default=0.5, ge=0.0)
    hard_example_conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ArduinoConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    transport: Literal["serial", "udp"] = "serial"
    port: str = ""                                  # COM/tty for the serial transport
    baud: int = Field(default=115200, ge=1200)      # ignored on R4 native USB; for bridges
    host: str = ""                                  # IP for the UDP/WiFi transport
    udp_port: int = Field(default=0, ge=0, le=65535)


class InputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    mouse_driver: Literal["sendinput", "arduino"] = "sendinput"
    # Compensate for the Windows pointer-speed slider on SendInput moves. Leave
    # OFF for games that read RAW input (most FPS) — they bypass pointer
    # ballistics, so compensating would over-move. ON only for cursor-driven games.
    compensate_ballistics: bool = False


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Hotkeys for the Calibrate tab so a 360° turn can be measured entirely
    # in-game (no GUI->game mouse travel corrupting the count).
    reset_key: str = "VK_HOME"                    # zero the raw-count total
    apply_key: str = "VK_END"                     # set sensitivity from the count


class OverlayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    show_confidence: bool = True                 # draw the detection score by each marker
    show_fov: bool = True                        # draw the FOV brackets
    show_boxes: bool = True                       # draw teammate/unknown do-not-shoot boxes
    show_tracking_line: bool = True              # dashed crosshair -> locked-target line
    diamond_scale: float = Field(default=1.0, gt=0.0, le=5.0)


class DynamicRoiConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    track_roi_size: int = Field(default=192, ge=32)
    model_input_px: int = Field(default=384, ge=64)
    max_missed_frames: int = Field(default=5, ge=1)
    rescan_interval_frames: int = Field(default=30, ge=0)   # 0 = no periodic rescan


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    capture: CaptureConfig = CaptureConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    classification: ClassificationConfig = ClassificationConfig()
    aim: AimConfig = AimConfig()
    motion: MotionConfig = MotionConfig()
    recoil: RecoilConfig = RecoilConfig()
    trigger: TriggerConfig = TriggerConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    training: TrainingConfig = TrainingConfig()
    arduino: ArduinoConfig = ArduinoConfig()
    input: InputConfig = InputConfig()
    overlay: OverlayConfig = OverlayConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    dynamic_roi: DynamicRoiConfig = DynamicRoiConfig()
