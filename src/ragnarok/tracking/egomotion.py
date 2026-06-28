"""Ego-motion (camera/global motion) providers.

The tracker consumes a 2x3 affine warp per frame. Phase 2 uses identity (no
compensation); Phase 3/4 can drop in a feed-forward GMC provider that returns a
real estimate without changing the tracker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EgoMotion(ABC):
    """Estimates a 2x3 affine warp mapping previous-frame coords to current."""

    @abstractmethod
    def estimate(self, frame) -> np.ndarray:
        """Return a (2, 3) float32 affine for ``frame``."""
        raise NotImplementedError


class IdentityEgoMotion(EgoMotion):
    """No-op provider: always returns the identity affine."""

    def estimate(self, frame) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)
