"""TensorRT engine detector (spec §5.2, §12.4).

detect() maps an injected Session's raw (boxes, scores, classes) into Detections,
fully unit-testable with a fake session. The real TensorRT engine load + inference
is lazy and box-only (needs the .engine + tensorrt + CUDA).
"""
from __future__ import annotations

from typing import Protocol

import cv2

from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector


class Session(Protocol):
    def infer(self, image, *, threshold: float) -> tuple[list, list, list]: ...


class RFDETRTensorRTDetector(Detector):
    def __init__(self, config: DetectionConfig, *, session: Session | None = None) -> None:
        self._config = config
        if session is None:
            session = _build_trt_session(config.engine_path)  # lazy/box-only
        self._session = session

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = self._session.infer(rgb, threshold=self._config.confidence)
        items = tuple(
            Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)),
                      confidence=float(s), class_id=int(c))
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes)
        )
        return Detections(items=items)


def _build_trt_session(engine_path: str) -> Session:  # pragma: no cover — box-only
    """Load a TensorRT engine into an inference Session. Box-only (tensorrt + CUDA)."""
    raise NotImplementedError(
        "Real TensorRT session loading is box-only; inject a Session in tests / "
        "implement the tensorrt runtime adapter on the deployment box "
        f"(engine_path={engine_path!r})."
    )
