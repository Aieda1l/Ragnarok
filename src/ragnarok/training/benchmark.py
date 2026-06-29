"""Benchmark harness: measure a Detector on a labeled frame set (spec §12.5, §15).

Aggregates single-class AP@iou_thresh + matched center-error across all images,
and reports latency p50/p99. CI-safe: inject a fake Detector + clock + synthetic
frames; real engines over real frames are a box-only smoke.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragnarok.core.clock import now_ns
from ragnarok.training.metrics import average_precision_at_iou, center_error


@dataclass(frozen=True)
class BenchmarkResult:
    map75: float
    center_error: float | None
    latency_p50_ms: float
    latency_p99_ms: float
    n_images: int


def run_benchmark(detector, dataset, *, clock=now_ns, iou_thresh: float = 0.75) -> BenchmarkResult:
    all_preds: list[tuple[tuple, float]] = []
    all_gts: list[tuple] = []
    latencies_ns: list[int] = []
    for frame, gt_boxes in dataset:
        t0 = clock()
        dets = detector.detect(frame)
        t1 = clock()
        latencies_ns.append(int(t1 - t0))
        all_preds.extend((d.xyxy, d.confidence) for d in dets)
        all_gts.extend(tuple(b) for b in gt_boxes)
    map75 = average_precision_at_iou(all_preds, all_gts, iou_thresh=iou_thresh)
    cerr = center_error(all_preds, all_gts)
    if latencies_ns:
        arr = np.asarray(latencies_ns, dtype=np.int64)
        p50, p99 = (float(v) / 1e6 for v in np.percentile(arr, [50, 99]))
    else:
        p50 = p99 = 0.0
    return BenchmarkResult(map75=map75, center_error=cerr,
                           latency_p50_ms=p50, latency_p99_ms=p99,
                           n_images=len(dataset))
