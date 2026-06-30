"""Tests for the dynamic-ROI SEARCH/TRACK FSM."""
from __future__ import annotations
from ragnarok.detection.roi import RoiMode, RoiState


def test_starts_in_search():
    assert RoiState(max_missed=3, rescan_interval=30).mode == RoiMode.SEARCH


def test_lock_enters_track():
    s = RoiState(max_missed=3, rescan_interval=30)
    assert s.update(has_lock=True) == RoiMode.TRACK
    assert s.mode == RoiMode.TRACK


def test_reverts_to_search_after_max_missed():
    s = RoiState(max_missed=2, rescan_interval=0)
    s.update(has_lock=True)               # TRACK
    assert s.update(has_lock=False) == RoiMode.TRACK   # 1 missed, still tracking
    assert s.update(has_lock=False) == RoiMode.SEARCH  # 2 missed -> SEARCH


def test_missed_counter_resets_on_relock():
    s = RoiState(max_missed=2, rescan_interval=0)
    s.update(has_lock=True)
    s.update(has_lock=False)              # 1 missed
    s.update(has_lock=True)               # relock -> reset
    assert s.update(has_lock=False) == RoiMode.TRACK   # only 1 missed again


def test_rescan_only_in_track_on_interval():
    s = RoiState(max_missed=3, rescan_interval=10)
    assert s.wants_rescan(10) is False    # still SEARCH -> no rescan concept
    s.update(has_lock=True)               # TRACK
    assert s.wants_rescan(10) is True     # 10 % 10 == 0
    assert s.wants_rescan(13) is False
    assert s.mode == RoiMode.TRACK        # rescan query does not change mode


def test_rescan_disabled_when_interval_zero():
    s = RoiState(max_missed=3, rescan_interval=0)
    s.update(has_lock=True)
    assert s.wants_rescan(0) is False
