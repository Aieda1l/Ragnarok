from __future__ import annotations
import warnings
import cv2
from ragnarok.core.types import Frame, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector, to_detections

_MODEL_CLASSES = {
    "nano": "RFDETRNano", "small": "RFDETRSmall",
    "medium": "RFDETRMedium", "large": "RFDETRLarge",
}


def _optimize_fp16(model, dtype=None) -> bool:
    """Fuse + FP16 the model for inference (rfdetr ``optimize_for_inference``).

    On Ampere this is roughly an order-of-magnitude speedup vs unoptimized fp32
    torch. Best-effort: rfdetr versions vary and some builds lack the method, so
    a failure warns and leaves the (slower but functional) model in place rather
    than breaking detection. ``dtype`` defaults to ``torch.float16`` via a lazy
    import so unit tests can pass a sentinel and avoid importing torch.
    """
    if dtype is None:
        import torch  # lazy: keeps torch out of unit tests
        dtype = torch.float16
    try:
        model.optimize_for_inference(dtype=dtype)
        return True
    except Exception as e:  # noqa: BLE001 — optimization must never break detection
        warnings.warn(f"optimize_for_inference failed ({e}); running unoptimized")
        return False


class RFDETRTorchDetector(Detector):
    def __init__(self, config: DetectionConfig, *, model=None) -> None:
        self._config = config
        if model is None:
            import rfdetr  # lazy: keeps torch/weights out of unit tests
            model = getattr(rfdetr, _MODEL_CLASSES[config.model])()
            if config.optimize_fp16:        # only auto-built models; never an injected one
                _optimize_fp16(model)
        self._model = model

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        sv = self._model.predict(rgb, threshold=self._config.confidence)
        return to_detections(sv)
