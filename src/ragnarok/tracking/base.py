"""Tracker abstraction and a no-op IdentityTracker.

Downstream code (worker loop, telemetry) depends only on this module, never on
the concrete vendored BoT-SORT wrapper, so the loop stays importable and
testable without numpy/scipy assignment machinery.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ragnarok.core.types import Detections, Track, Tracks

# Injected ego-motion default: identity (no global-motion compensation).
IDENTITY_AFFINE = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


class Tracker(ABC):
    """Maps detections to tracks with stable ids across frames."""

    @abstractmethod
    def update(self, detections: Detections, ego_affine=IDENTITY_AFFINE) -> Tracks:
        """Advance one frame and return the current confirmed tracks."""
        raise NotImplementedError


class IdentityTracker(Tracker):
    """No-op tracker: echoes each detection as a fresh track.

    Assigns a new incrementing id to every detection on every call (no temporal
    association). Used as the default in the worker loop and in tests so the
    pipeline runs without the vendored tracker.
    """

    def __init__(self) -> None:
        self._next_id = 0

    def update(self, detections: Detections, ego_affine=IDENTITY_AFFINE) -> Tracks:
        tracks = []
        for det in detections:
            self._next_id += 1
            tracks.append(Track.from_detection(det, self._next_id))
        return Tracks(items=tuple(tracks))
