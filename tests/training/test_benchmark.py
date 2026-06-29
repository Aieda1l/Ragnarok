"""Tests for the benchmark harness against a fake detector (no GPU/frames)."""
from __future__ import annotations
import numpy as np
from ragnarok.core.types import Frame, Detection, Detections
from ragnarok.training.benchmark import run_benchmark, BenchmarkResult


def _frame():
    return Frame(np.zeros((384, 384, 3), np.uint8), t_capture_ns=1, region=(0, 0, 384, 384))


class _PerfectDetector:
    """Returns exactly the gt box (passed via a side channel) with conf 0.9."""
    def __init__(self, boxes):
        self._boxes = boxes
    def detect(self, frame):
        return Detections(items=tuple(
            Detection(xyxy=b, confidence=0.9, class_id=0) for b in self._boxes))


class _StepClock:
    def __init__(self, dt_ns):
        self.t = 0
        self._dt = dt_ns
    def __call__(self):
        v = self.t
        self.t += self._dt
        return v


def test_perfect_detector_scores_map_1():
    gt = [(0.0, 0.0, 10.0, 10.0)]
    dataset = [(_frame(), gt)]
    res = run_benchmark(_PerfectDetector(gt), dataset, clock=_StepClock(2_000_000))
    assert isinstance(res, BenchmarkResult)
    assert res.map75 == 1.0
    assert res.center_error == 0.0
    assert res.n_images == 1
    assert res.latency_p50_ms > 0.0


def test_aggregates_across_images():
    g1 = [(0.0, 0.0, 10.0, 10.0)]
    g2 = [(20.0, 20.0, 30.0, 30.0)]
    dataset = [(_frame(), g1), (_frame(), g2)]
    # detector returns g1 for both images -> image 2 is a miss
    res = run_benchmark(_PerfectDetector(g1), dataset, clock=_StepClock(1_000_000))
    assert res.n_images == 2
    assert 0.0 <= res.map75 <= 1.0
