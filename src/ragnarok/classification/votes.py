"""Per-track temporal vote: a track is ENEMY only after >= min_agree of the
last `window` per-frame decisions are True. Stabilizes one-frame mislabels."""
from __future__ import annotations
from collections import defaultdict, deque


class TrackVoteBook:
    def __init__(self, window: int = 5, min_agree: int = 3) -> None:
        self.window = window
        self.min_agree = min_agree
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, frame_decision: bool) -> bool:
        h = self._hist[track_id]
        h.append(bool(frame_decision))
        return bool(sum(h) >= self.min_agree)

    def label(self, track_id: int) -> str:
        return "enemy" if sum(self._hist.get(track_id, ())) >= self.min_agree else "unknown"

    def prune(self, live_ids) -> None:
        for tid in [t for t in self._hist if t not in live_ids]:
            del self._hist[tid]
