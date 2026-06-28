from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ragnarok.core.types import Track

@dataclass(frozen=True)
class TelemetrySnapshot:
    fps: float
    loop_ms_p50: float
    loop_ms_p99: float
    detection_count: int
    preview: np.ndarray | None   # small BGR image for the GUI, or None
    seq: int
    tracks: tuple[Track, ...] = ()  # Phase 2; default keeps Phase 1 callers working

class SnapshotPublisher:
    """Single-writer (worker) / single-reader (GUI). publish() rebinds one
    attribute -> GIL-atomic; the reader gets a whole snapshot or None, never torn."""
    def __init__(self) -> None:
        self._latest: TelemetrySnapshot | None = None

    def publish(self, snap: TelemetrySnapshot) -> None:
        self._latest = snap

    def latest(self) -> TelemetrySnapshot | None:
        return self._latest
