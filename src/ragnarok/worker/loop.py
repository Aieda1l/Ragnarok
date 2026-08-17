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
from ragnarok.aim.latency_measure import WallLatencyMeasurer

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
        self._retired: list = []            # controllers swapped out, released on the worker thread
        self._seq = 0
        self._last_ns: int | None = None
        self._measure_mouse = None          # SendInput driver for latency measurement
        self._measure_req: float | None = None   # requested duration_s (GIL-atomic rebind)
        self._measure_ms: float | None = None    # result, surfaced in ONE snapshot

    def set_detector(self, detector) -> None:
        """Atomically hot-swap the detector (box-only rebuild for backend/engine)."""
        self._det = detector

    def set_detector_confidence(self, conf: float) -> None:
        """Cheap live threshold update (no model rebuild)."""
        self._det.set_confidence(conf)

    def set_measure_mouse(self, mouse) -> None:
        self._measure_mouse = mouse

    def stop_capture(self) -> None:
        self._cap.stop()

    def request_latency_measure(self, duration_s: float = 2.5) -> None:
        self._measure_req = float(duration_s)    # consumed once at the top of tick()
        self._measure_ms = None                  # reset the latch for the fresh measurement

    def set_aim_controller(self, controller) -> None:
        """Atomically hot-swap the aim controller (or None to disable aim).

        Single attribute rebind -> GIL-atomic; the tick loop reads self._aim
        once per iteration, so it always sees a whole controller or None. The
        outgoing controller is queued for release on the WORKER thread (next
        tick), not released here on the GUI thread: releasing on the GUI thread
        could interleave with the worker's in-flight TriggerBot.update and leave
        a stuck press. Draining on the worker thread strictly orders the release
        after any in-flight press.
        """
        prev = self._aim
        self._aim = controller
        if prev is not None and prev is not controller:
            self._retired.append(prev)

    def set_tracker(self, tracker) -> None:
        """Atomically hot-swap the tracker (None restores the identity default).

        The tick reads self._tracker exactly once per iteration, so a rebind is
        race-free (a tick sees a whole tracker, never a partial one)."""
        self._tracker = tracker if tracker is not None else IdentityTracker()

    def set_classifier(self, classifier) -> None:
        """Atomically hot-swap the friend/foe classifier (None restores null)."""
        self._classifier = classifier if classifier is not None else NullClassifier()

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, self._preview_max / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return np.ascontiguousarray(image)

    def tick(self) -> None:
        # Release controllers swapped out since the last tick (on the worker thread,
        # so a released button can't race an in-flight press). Shared mouse -> a
        # single release un-sticks the physical button.
        while self._retired:
            r = self._retired.pop()
            rel = getattr(r, "release", None)
            if callable(rel):
                try:
                    rel()
                except Exception as e:
                    import warnings
                    warnings.warn(f"retired controller release failed: {e}")
        req = self._measure_req
        if req is not None:                      # latency measurement: blocks this tick
            self._measure_req = None
            self._measure_ms = None
            if self._measure_mouse is not None:
                lag = WallLatencyMeasurer(self._cap, self._measure_mouse, duration_s=req).run()
                self._measure_ms = round(lag * 1000.0, 1) if lag is not None else None
        # Snapshot the detector ONCE: the GUI thread may hot-swap self._det (e.g.
        # a DynamicRoiDetector -> plain detector) at any moment, so reading it for
        # detect() and for the observe_lock feature-check separately would be a
        # TOCTOU race (the second read could miss observe_lock -> AttributeError
        # kills the worker thread). Same pattern as the aim snapshot below.
        det = self._det
        t0 = now_ns()
        frame = self._cap.grab()
        t_cap = now_ns()
        if frame is None:
            return
        dets = det.detect(frame)
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

        # Effective lock target for the overlay + dynamic-ROI: the aim lock, or the
        # trigger's crosshair-target when aim assist is off (trigger-only mode), so
        # both keep tracking the enemy under the crosshair either way.
        lock_id = None
        if aim is not None:
            lock_id = getattr(aim, "target_id", None)
            if lock_id is None:
                lock_id = getattr(aim, "fire_target_id", None)

        # Dynamic-ROI feedback: tell the detector where the locked target is so the
        # NEXT frame can crop+upscale around it (no-op for the plain detector).
        if hasattr(det, "observe_lock"):
            locked = next((t for t in tracks if t.track_id == lock_id), None) if lock_id is not None else None
            det.observe_lock(locked.center if locked is not None else None,
                             locked is not None)

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
            locked_target_id=lock_id,
            roi_region=frame.region,
            latency_ms=self._measure_ms,
            aim_on=getattr(aim, "aim_on", None),
            trigger_on=getattr(aim, "trigger_on", None),
        ))
        # NOTE: latency_ms is LATCHED — it persists in every snapshot until the next
        # request_latency_measure resets it. The Calibrate panel polls at 200 ms, so a
        # one-tick lifetime (the old behaviour) was caught only ~3% of the time.

    def run(self, stop_event: threading.Event) -> None:
        self._cap.start()
        try:
            while not stop_event.is_set():
                try:
                    self.tick()
                except Exception:                # a hot-swap race / transient error
                    import time
                    import traceback
                    import warnings
                    warnings.warn("worker tick failed:\n" + traceback.format_exc())
                    time.sleep(0.05)
        finally:
            self._cap.stop()
