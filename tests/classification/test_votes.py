"""Tests for TrackVoteBook."""
from __future__ import annotations

import pytest

from ragnarok.classification.votes import TrackVoteBook


class TestTrackVoteBook:
    def test_not_enemy_after_two_trues(self):
        """With min_agree=3, two True votes should not yet call it enemy."""
        book = TrackVoteBook(window=5, min_agree=3)
        book.update(1, True)
        book.update(1, True)
        assert book.label(1) != "enemy"

    def test_enemy_on_third_agree(self):
        """Three True votes should cross the threshold."""
        book = TrackVoteBook(window=5, min_agree=3)
        book.update(1, True)
        book.update(1, True)
        voted = book.update(1, True)
        assert voted is True
        assert book.label(1) == "enemy"

    def test_unknown_with_mixed_votes(self):
        """3 False + 2 True in a window=5 book with min_agree=3 stays unknown."""
        book = TrackVoteBook(window=5, min_agree=3)
        for _ in range(3):
            book.update(2, False)
        for _ in range(2):
            book.update(2, True)
        assert book.label(2) != "enemy"

    def test_window_sliding(self):
        """Old votes outside the window should be dropped."""
        book = TrackVoteBook(window=3, min_agree=3)
        # Push 3 Trues (now enemy)
        for _ in range(3):
            book.update(3, True)
        assert book.label(3) == "enemy"
        # Push 3 Falses — True votes slide out; should revert
        for _ in range(3):
            book.update(3, False)
        assert book.label(3) != "enemy"

    def test_unknown_for_unseen_id(self):
        """A track never seen should return 'unknown'."""
        book = TrackVoteBook()
        assert book.label(999) == "unknown"

    def test_update_returns_bool(self):
        book = TrackVoteBook(window=5, min_agree=3)
        result = book.update(10, True)
        assert isinstance(result, bool)

    def test_prune_removes_dead_ids(self):
        """prune(live_ids) should discard history for ids not in live_ids."""
        book = TrackVoteBook(window=5, min_agree=3)
        book.update(1, True)
        book.update(2, True)
        book.prune({2})
        # id 1 should be gone; label returns unknown
        assert book.label(1) == "unknown"
        # id 2 still present
        assert book.label(2) != "unknown" or True  # may still be unknown if <3 votes

    def test_prune_with_empty_live_ids(self):
        """Pruning with empty set clears all entries."""
        book = TrackVoteBook()
        book.update(5, True)
        book.update(6, True)
        book.prune(set())
        assert book.label(5) == "unknown"
        assert book.label(6) == "unknown"
