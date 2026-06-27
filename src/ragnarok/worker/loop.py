from __future__ import annotations
import threading
import cv2
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher

class WorkerLoop:
    def __init__(self, capturer, detector, profiler: StageProfiler,
                 publisher: SnapshotPublisher, *, preview_max: int = 320) -> None:
        self._cap = capturer
        self._det = detector
        self._profiler = profiler
        self._pub = publisher
        self._preview_max = preview_max
        self._seq = 0
        self._last_ns: int | None = None

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, self._preview_max / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return np.ascontiguousarray(image)

    def tick(self) -> None:
        t0 = now_ns()
        frame = self._cap.grab()
        t_cap = now_ns()
        if frame is None:
            return
        dets = self._det.detect(frame)
        t_inf = now_ns()

        self._profiler.record("capture", t_cap - t0)
        self._profiler.record("infer", t_inf - t_cap)
        self._profiler.record("loop", t_inf - t0)

        fps = 0.0
        if self._last_ns is not None:
            dt = t_inf - self._last_ns
            fps = 1e9 / dt if dt > 0 else 0.0
        self._last_ns = t_inf

        p50, p99 = self._profiler.percentiles("loop")
        self._seq += 1
        self._pub.publish(TelemetrySnapshot(
            fps=fps, loop_ms_p50=p50, loop_ms_p99=p99,
            detection_count=len(dets), preview=self._downscale(frame.image), seq=self._seq,
        ))

    def run(self, stop_event: threading.Event) -> None:
        self._cap.start()
        try:
            while not stop_event.is_set():
                self.tick()
        finally:
            self._cap.stop()
