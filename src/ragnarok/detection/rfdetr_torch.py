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


def _report_device() -> bool:
    """Warn loudly if inference will run on the CPU (seconds/frame) instead of
    CUDA. Returns True when CUDA is available. Best-effort; never raises."""
    try:
        import torch  # lazy: keeps torch out of unit tests
    except Exception:  # noqa: BLE001
        return False
    if torch.cuda.is_available():
        try:
            warnings.warn(f"[detector] CUDA OK: {torch.cuda.get_device_name(0)}")
        except Exception:  # noqa: BLE001
            warnings.warn("[detector] CUDA available")
        return True
    warnings.warn(
        "[detector] CUDA is NOT available — RF-DETR will run on the CPU "
        "(~seconds per frame, ~0.2 FPS). Install a CUDA torch build matching "
        "your GPU/driver (e.g. pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121) so it uses the RTX 3090.")
    return False


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
        self._confidence = float(config.confidence)   # live-tunable threshold
        if model is None:
            import rfdetr  # lazy: keeps torch/weights out of unit tests
            on_cuda = _report_device()      # loud warning if this will run on CPU
            model = getattr(rfdetr, _MODEL_CLASSES[config.model])()
            # FP16 optimize only helps (and only works) on CUDA; skip on CPU.
            if config.optimize_fp16 and on_cuda:  # only auto-built models; never injected
                _optimize_fp16(model)
        self._model = model

    def set_confidence(self, conf: float) -> None:
        self._confidence = float(conf)

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        sv = self._model.predict(rgb, threshold=self._confidence)
        return to_detections(sv)
