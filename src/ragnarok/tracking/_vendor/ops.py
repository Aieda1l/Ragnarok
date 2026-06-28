"""Bounding-box format conversions.

Derived from the static conversion helpers in NirAharon/BoT-SORT
(tracker/bot_sort.py STrack), MIT licensed. numpy only.

Formats:
  tlwh -> (top-left x, top-left y, width, height)
  tlbr -> (x1, y1, x2, y2)
  xywh -> (center x, center y, width, height)
"""
import numpy as np


def tlwh_to_xywh(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret


def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def xywh_to_tlwh(xywh):
    ret = np.asarray(xywh).copy()
    ret[:2] -= ret[2:] / 2
    return ret
