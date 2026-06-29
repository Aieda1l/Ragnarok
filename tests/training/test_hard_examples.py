"""Tests for pure hard-example selection."""
from __future__ import annotations
from ragnarok.training.hard_examples import select_hard_examples


def test_selects_low_confidence_and_missed():
    records = [("a", 0.95), ("b", 0.30), ("c", None), ("d", 0.49)]
    assert select_hard_examples(records, conf_threshold=0.5) == ["b", "c", "d"]


def test_empty_when_all_confident():
    records = [("a", 0.9), ("b", 0.8)]
    assert select_hard_examples(records, conf_threshold=0.5) == []


def test_preserves_input_order():
    records = [("z", None), ("y", 0.1), ("x", 0.99)]
    assert select_hard_examples(records, conf_threshold=0.5) == ["z", "y"]
