from __future__ import annotations
import threading
import cv2
import numpy as np
from ragnarok.core.clock import now_ns
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.tracking.base import Tracker, IdentityTracker
from ragnarok.classification.base import FriendFoeClassifier, NullClassifier
from ragnarok.gui.overlay import draw_overlay

class WorkerLoop:
    def __init__(self, capturer, detector, profiler: StageProfiler,
                 publisher: SnapshotPublisher, *, preview_max: int = 320,
                 tracker: Tracker | None = None,
                 classifier: FriendFoeClassifier | None = None,
                 aim_controller=None) -> None:
        self._cap = capturer
        self._det = detector
        self._profiler = profiler
        self._pub = publisher
        self._preview_max = preview_max
        self._tracker = tracker or IdentityTracker()         # defaults keep Phase 1 tests passing
        self._classifier = classifier or NullClassifier()
        self._aim = aim_controller          # optional; None keeps Phase 1/2 behavior
        self._seq = 0
        self._last_ns: int | None = None

    def set_aim_controller(self, controller) -> None:
        """Atomically hot-swap the aim controller (or None to disable aim).

        Single attribute rebind -> GIL-atomic; the tick loop reads self._aim
        once per iteration, so it always sees a whole controller or None.
        """
        self._aim = controller

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
        tracks = self._tracker.update(dets, frame)   # frame carries t_capture_ns for GMC
        t_trk = now_ns()
        tracks = self._classifier.classify(tracks, frame.image)
        t_cls = now_ns()

        # Snapshot the controller ONCE: the GUI thread may hot-swap self._aim
        # (incl. to None on a live disable) at any moment, so a fresh load for
        # the guard vs. the call would be a TOCTOU race (None.update()).
        aim = self._aim
        if aim is not None:
            aim.update(tracks, frame.t_capture_ns)
        t_aim = now_ns()

        self._profiler.record("capture", t_cap - t0)
        self._profiler.record("infer", t_inf - t_cap)
        self._profiler.record("track", t_trk - t_inf)
        self._profiler.record("classify", t_cls - t_trk)
        self._profiler.record("aim", t_aim - t_cls)
        self._profiler.record("loop", t_aim - t0)

        fps = 0.0
        if self._last_ns is not None:
            dt = t_aim - self._last_ns
            fps = 1e9 / dt if dt > 0 else 0.0
        self._last_ns = t_aim

        p50, p99 = self._profiler.percentiles("loop")
        self._seq += 1
        h, w = frame.image.shape[:2]
        scale = min(1.0, self._preview_max / max(h, w))
        preview = self._downscale(frame.image)
        draw_overlay(preview, tracks, scale)
        self._pub.publish(TelemetrySnapshot(
            fps=fps, loop_ms_p50=p50, loop_ms_p99=p99,
            detection_count=len(dets), preview=preview, seq=self._seq,
            tracks=tuple(tracks),
            locked_target_id=getattr(aim, "target_id", None),
            roi_region=frame.region,
        ))

    def run(self, stop_event: threading.Event) -> None:
        self._cap.start()
        try:
            while not stop_event.is_set():
                self.tick()
        finally:
            self._cap.stop()
