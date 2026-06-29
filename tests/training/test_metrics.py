"""Tests for pure detection metrics against analytic cases."""
from __future__ import annotations
import math
from ragnarok.training.metrics import iou, average_precision_at_iou, center_error


def test_iou_half_overlap():
    a = (0.0, 0.0, 2.0, 2.0)        # area 4
    b = (1.0, 0.0, 3.0, 2.0)        # area 4, overlap x in [1,2] -> 2*2=2... overlap area = 1*2 = 2
    # intersection = 2, union = 4+4-2 = 6 -> 1/3
    assert abs(iou(a, b) - (2.0 / 6.0)) < 1e-9


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_ap_perfect_detection():
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9)]   # exact match, IoU 1.0
    assert average_precision_at_iou(preds, gts, iou_thresh=0.75) == 1.0


def test_ap_low_iou_is_miss():
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((5.0, 5.0, 15.0, 15.0), 0.9)]   # IoU well below 0.75
    assert average_precision_at_iou(preds, gts, iou_thresh=0.75) == 0.0


def test_ap_half_recall_is_half():
    # two gt, one matched -> recall caps at 0.5, precision 1.0 -> AP 0.5
    gts = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9)]
    assert abs(average_precision_at_iou(preds, gts, iou_thresh=0.75) - 0.5) < 1e-9


def test_ap_extra_false_positive_does_not_reduce_perfect_recall():
    # one tp (high score) + one fp (low score) vs one gt -> envelope keeps AP 1.0
    gts = [(0.0, 0.0, 10.0, 10.0)]
    preds = [((0.0, 0.0, 10.0, 10.0), 0.9), ((50.0, 50.0, 60.0, 60.0), 0.1)]
    assert abs(average_precision_at_iou(preds, gts, iou_thresh=0.75) - 1.0) < 1e-9


def test_ap_no_gt_no_preds_is_one():
    assert average_precision_at_iou([], [], iou_thresh=0.75) == 1.0


def test_ap_no_gt_with_preds_is_zero():
    assert average_precision_at_iou([((0, 0, 1, 1), 0.9)], [], iou_thresh=0.75) == 0.0


def test_center_error_matched_distance():
    gts = [(0.0, 0.0, 10.0, 10.0)]                 # center (5,5)
    preds = [(0.0, 2.0, 10.0, 12.0)]               # center (5,7), IoU 0.667 >= 0.5 -> matches; dist 2
    assert abs(center_error(preds, gts) - 2.0) < 1e-9


def test_center_error_none_when_no_match():
    assert center_error([(0, 0, 1, 1)], [(100, 100, 110, 110)]) is None
