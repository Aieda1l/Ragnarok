from __future__ import annotations
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector


def create_detector(config: DetectionConfig) -> Detector:
    if config.backend == "rfdetr_trt":
        from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector
        return RFDETRTensorRTDetector(config)
    from ragnarok.detection.rfdetr_torch import RFDETRTorchDetector
    return RFDETRTorchDetector(config)
