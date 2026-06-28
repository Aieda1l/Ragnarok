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
    backend: Literal["rfdetr_torch"] = "rfdetr_torch"
    model: Literal["nano", "small", "medium", "large"] = "small"  # Apache-2.0 variants only
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


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
    aimer: Literal["flick", "feedback"] = "feedback"
    kp: float = Field(default=0.35, gt=0.0, le=2.0)
    max_step_px: float = Field(default=60.0, gt=0.0)
    flick_speed_px_s: float = Field(default=4000.0, gt=0.0)
    ema_alpha: float = Field(default=0.5, gt=0.0, le=1.0)
    aim_point: Literal["head", "body"] = "head"
    head_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    sensitivity: float = Field(default=0.022, gt=0.0)              # deg per mouse count
    lead_ms: float = Field(default=40.0, ge=0.0, le=500.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    capture: CaptureConfig = CaptureConfig()
    detection: DetectionConfig = DetectionConfig()
    aim: AimConfig = AimConfig()
