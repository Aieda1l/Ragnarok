"""TensorRT engine detector (spec §5.2, §12.4).

detect() maps an injected Session's raw (boxes, scores, classes) into Detections,
fully unit-testable with a fake session. The real TensorRT engine load + inference
is lazy and box-only (needs the .engine + tensorrt + torch/CUDA + rfdetr).

The runtime reuses rfdetr's own ``PostProcess`` (exact box decoding) and rfdetr's
ImageNet normalization / 512² resolution, so TRT detections match the torch path.
Build an engine with ``build_trt_engine(onnx_path, engine_path)`` (or the trtexec
tooling in detection.export).
"""
from __future__ import annotations

from typing import Protocol

import cv2

from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.config.schema import DetectionConfig
from ragnarok.detection.base import Detector

# RF-DETR (Small) inference constants — must match the exported model.
_RESOLUTION = 512
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Session(Protocol):
    def infer(self, image, *, threshold: float) -> tuple[list, list, list]: ...


class RFDETRTensorRTDetector(Detector):
    def __init__(self, config: DetectionConfig, *, session: Session | None = None) -> None:
        self._config = config
        self._confidence = float(config.confidence)   # live-tunable threshold
        if session is None:
            session = _build_trt_session(config.engine_path)  # lazy/box-only
        self._session = session

    def set_confidence(self, conf: float) -> None:
        self._confidence = float(conf)

    def detect(self, frame: Frame) -> Detections:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = self._session.infer(rgb, threshold=self._confidence)
        items = tuple(
            Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)),
                      confidence=float(s), class_id=int(c))
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes)
        )
        return Detections(items=items)


class _TensorRTSession:  # pragma: no cover — box-only (tensorrt + torch/CUDA + rfdetr)
    """Runs a deserialized RF-DETR TensorRT engine and postprocesses via rfdetr.

    Buffers are preallocated as torch CUDA tensors and bound once; each infer()
    resizes+normalizes the ROI, runs the engine, and decodes with rfdetr's
    ``PostProcess`` into ROI-pixel ``(boxes, scores, classes)``.
    """

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt
        import torch
        from rfdetr.inference import PostProcess

        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        # Allocate one CUDA buffer per engine I/O tensor from its declared shape,
        # so the runtime adapts to the model's class count (91 for the base model,
        # N for a trained model) and input resolution automatically.
        self._buf = {}
        self._input = self._box = self._lbl = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            self._buf[name] = torch.zeros(shape, device="cuda")
            self._ctx.set_tensor_address(name, self._buf[name].data_ptr())
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input = name
            elif shape[-1] == 4:            # boxes tensor -> last dim 4
                self._box = name
            else:                           # class logits tensor
                self._lbl = name
        _, _, self._in_h, self._in_w = self._buf[self._input].shape
        self._stream = torch.cuda.Stream()
        self._mean = torch.tensor(_MEAN, device="cuda").view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device="cuda").view(1, 3, 1, 1)
        self._post = PostProcess(num_select=300)

    def infer(self, rgb, *, threshold: float):
        torch = self._torch
        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (self._in_w, self._in_h))
        t = torch.from_numpy(img).to("cuda").float().permute(2, 0, 1).unsqueeze(0) / 255.0
        self._buf[self._input].copy_((t - self._mean) / self._std)
        self._ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        outputs = {"pred_logits": self._buf[self._lbl], "pred_boxes": self._buf[self._box]}
        target = torch.tensor([[h, w]], device="cuda")
        res = self._post(outputs, target)[0]
        keep = res["scores"] >= threshold
        boxes = res["boxes"][keep].cpu().tolist()
        scores = res["scores"][keep].cpu().tolist()
        classes = res["labels"][keep].cpu().tolist()
        return (boxes, scores, classes)


def _build_trt_session(engine_path: str) -> Session:  # pragma: no cover — box-only
    """Load a TensorRT engine into an inference Session (box-only: tensorrt + CUDA)."""
    if not engine_path:
        raise RuntimeError(
            "detection.engine_path is empty; build an engine "
            "(build_trt_engine) and set engine_path for the rfdetr_trt backend.")
    return _TensorRTSession(engine_path)


def build_trt_engine(onnx_path: str, engine_path: str, *,
                     workspace_bytes: int = 1 << 31) -> str:  # pragma: no cover — box-only
    """Build a TensorRT engine from an ONNX file via the tensorrt Python API.

    TensorRT 11 auto-selects reduced-precision (TF32/FP16) tactics on Ampere+,
    so no explicit precision flag is set (the FP16 builder flag was removed).
    Returns ``engine_path``.
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    return engine_path
