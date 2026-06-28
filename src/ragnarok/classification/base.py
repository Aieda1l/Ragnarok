"""Friend/foe classifier: HSV outline-ring color gate + temporal vote.

classify(tracks, image) takes the BGR ndarray (for HSV ring sampling) and
returns a NEW Tracks with each track's team updated. Insufficient-vote tracks
stay Team.UNKNOWN (never auto-target)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import replace
import numpy as np

from ragnarok.core.types import Tracks, Team
from ragnarok.classification.color import ColorProfile, is_enemy_frame
from ragnarok.classification.votes import TrackVoteBook


class FriendFoeClassifier(ABC):
    @abstractmethod
    def classify(self, tracks: Tracks, image: np.ndarray) -> Tracks: ...


class NullClassifier(FriendFoeClassifier):
    """Default / test fake: pass tracks through unchanged."""
    def classify(self, tracks: Tracks, image: np.ndarray) -> Tracks:
        return tracks


class HSVRingClassifier(FriendFoeClassifier):
    def __init__(self, profile: ColorProfile, *, frac_threshold: float = 0.18,
                 thickness: int = 4, vote_window: int = 5, vote_min: int = 3) -> None:
        self._profile = profile
        self._frac_threshold = frac_threshold
        self._thickness = thickness
        self._votes = TrackVoteBook(window=vote_window, min_agree=vote_min)

    def classify(self, tracks: Tracks, image: np.ndarray) -> Tracks:
        out = []
        live: set[int] = set()
        for tr in tracks:
            live.add(tr.track_id)
            decision = is_enemy_frame(image, tr.xyxy, self._profile,
                                      frac_threshold=self._frac_threshold,
                                      thickness=self._thickness)
            is_enemy = self._votes.update(tr.track_id, decision)
            team = Team.ENEMY if is_enemy else Team.UNKNOWN
            out.append(replace(tr, team=team))
        self._votes.prune(live)
        return Tracks(items=tuple(out))
