from __future__ import annotations
from collections import defaultdict, deque
import numpy as np

class StageProfiler:
    def __init__(self, window: int = 240) -> None:
        self._window = window
        self._samples: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, stage: str, dt_ns: int) -> None:
        self._samples[stage].append(int(dt_ns))

    def percentiles(self, stage: str) -> tuple[float, float]:
        buf = self._samples.get(stage)
        if not buf:
            return (0.0, 0.0)
        arr = np.fromiter(buf, dtype=np.int64)
        p50, p99 = np.percentile(arr, [50, 99])
        return (float(p50) / 1e6, float(p99) / 1e6)

    def stages(self) -> list[str]:
        return list(self._samples.keys())
