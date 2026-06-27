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

class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    capture: CaptureConfig = CaptureConfig()
    detection: DetectionConfig = DetectionConfig()
