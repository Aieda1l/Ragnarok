"""BotSortTracker: a Tracker over the vendored motion-only BoT-SORT core.

Maps our ``Detections`` to the core's ``[x1, y1, x2, y2, conf, cls]`` array,
runs the two-stage association with an injected ego-motion provider, and emits
our ``Tracks`` (``team=Team.UNKNOWN`` — friend/foe is a later stage).
"""
from __future__ import annotations

import numpy as np

from ragnarok.core.types import Detections, Team, Track, Tracks

from ._vendor.botsort_core import BoTSORT
from .base import Tracker
from .egomotion import EgoMotion, IdentityEgoMotion


class BotSortTracker(Tracker):
    def __init__(self, *, ego: EgoMotion | None = None,
                 track_high_thresh: float = 0.6, track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.7, track_buffer: int = 30,
                 match_thresh: float = 0.8, proximity_thresh: float = 0.5,
                 frame_rate: int = 30) -> None:
        self.ego = ego if ego is not None else IdentityEgoMotion()
        self._core = BoTSORT(
            ego=self.ego,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            frame_rate=frame_rate,
        )

    def update(self, detections: Detections, frame=None) -> Tracks:
        rows = [
            [d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3], d.confidence, d.class_id]
            for d in detections
        ]
        output_results = (
            np.asarray(rows, dtype=float) if rows else np.empty((0, 6), dtype=float)
        )
        # The frame is consumed only by the injected ego provider
        # (IdentityEgoMotion ignores it; FeedForwardGMC reads frame.t_capture_ns).
        stracks = self._core.update(output_results, frame=frame)

        frame_id = self._core.frame_id
        tracks = []
        for st in stracks:
            x1, y1, x2, y2 = (float(v) for v in st.tlbr)
            tracks.append(Track(
                track_id=int(st.track_id),
                xyxy=(x1, y1, x2, y2),
                confidence=float(st.score),
                class_id=int(st.cls),
                team=Team.UNKNOWN,
                age=int(frame_id - st.start_frame),
                hits=int(st.tracklet_len + 1),
                time_since_update=int(frame_id - st.frame_id),
            ))
        return Tracks(items=tuple(tracks))
