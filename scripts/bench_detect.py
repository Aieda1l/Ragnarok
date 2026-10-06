"""Measure detect() latency of the TensorRT engine and write a JSON log.

Times ``RFDETRTensorRTDetector.detect`` end to end (BGR->RGB, resize/normalize,
engine, rfdetr PostProcess) on a fixed ROI-sized frame, after a warm-up.

Run:  uv run python scripts/bench_detect.py --engine engines/rfdetr-trained.engine
Out:  docs/results/detect_latency.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from ragnarok.config.schema import DetectionConfig
from ragnarok.core.types import Frame
from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector


def _frame(roi: int, image_dir: Path | None) -> tuple[Frame, str]:
    if image_dir is not None:
        for p in sorted(image_dir.glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (roi, roi))
                return Frame(image=img, t_capture_ns=0, region=(0, 0, roi, roi)), p.name
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (roi, roi, 3), dtype=np.uint8)
    return Frame(image=img, t_capture_ns=0, region=(0, 0, roi, roi)), "synthetic-noise"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engines/rfdetr-trained.engine")
    ap.add_argument("--roi", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--image-dir", type=Path, default=Path("dataset/valid"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/detect_latency.json"))
    a = ap.parse_args()

    import tensorrt
    import torch

    det = RFDETRTensorRTDetector(
        DetectionConfig(backend="rfdetr_trt", engine_path=a.engine))
    frame, src = _frame(a.roi, a.image_dir if a.image_dir.is_dir() else None)
    for _ in range(a.warmup):
        det.detect(frame)
    ms = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        det.detect(frame)
        ms.append((time.perf_counter() - t0) * 1e3)
    ms.sort()
    p50 = statistics.median(ms)
    result = {
        "engine": a.engine, "roi_px": a.roi, "frame_source": src,
        "warmup": a.warmup, "iters": a.iters,
        "latency_ms": {"mean": statistics.fmean(ms), "p50": p50,
                       "p99": ms[int(0.99 * len(ms)) - 1], "max": ms[-1]},
        "fps_from_p50": 1000.0 / p50,
        "fps_from_mean": 1000.0 / statistics.fmean(ms),
        "env": {"gpu": torch.cuda.get_device_name(0), "tensorrt": tensorrt.__version__,
                "torch": torch.__version__, "python": platform.python_version(),
                "os": platform.platform()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
