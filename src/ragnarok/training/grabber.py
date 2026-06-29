"""In-app smart frame grabber (spec §12 step 1).

Rate-limited; delegates the actual disk write to an injected `writer` callable so
the decision logic is unit-testable and CI never touches disk. Saves frames the
detector is uncertain about or that show a scene change (see sampling.py).
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.training.sampling import should_capture


class FrameGrabber:
    def __init__(self, *, writer, conf_threshold: float, scene_change_threshold: float,
                 min_interval_s: float, clock=now_ns) -> None:
        self._writer = writer
        self._conf = conf_threshold
        self._scene = scene_change_threshold
        self._min_interval_ns = int(min_interval_s * 1e9)
        self._clock = clock
        self._last_saved_image = None
        self._last_save_ns: int | None = None
        self.count = 0

    def offer(self, frame, detections) -> bool:
        now = self._clock()
        if self._last_save_ns is not None and now - self._last_save_ns < self._min_interval_ns:
            return False
        if not should_capture(detections, frame.image, self._last_saved_image,
                              conf_threshold=self._conf,
                              scene_change_threshold=self._scene):
            return False
        self._writer(frame.image, frame.t_capture_ns)
        # copy: frame.image may alias a reused capture ring-buffer the capture
        # thread overwrites; the retained last-saved image must be stable.
        self._last_saved_image = frame.image.copy()
        self._last_save_ns = now
        self.count += 1
        return True
