"""Tests for dynamic-ROI coordinate transforms (pure)."""
from __future__ import annotations
from ragnarok.detection.roi import letterbox_params, map_back_letterbox


def test_letterbox_square_source_is_uniform():
    # 384x384 source into 384 -> scale 1, no pad
    assert letterbox_params(384, 384, 384) == (1.0, 0.0, 0.0)


def test_letterbox_wide_source_pads_vertically():
    # 768x384 into 384 -> scale 0.5, scaled height 192, pad_y 96, pad_x 0
    scale, px, py = letterbox_params(768, 384, 384)
    assert scale == 0.5 and px == 0.0 and py == 96.0


def test_letterbox_roundtrip_inverse():
    # A box known in source space -> forward (scale+pad) -> map_back recovers it.
    scale, px, py = letterbox_params(768, 384, 384)        # scale 0.5, py 96
    src_box = (100.0, 50.0, 200.0, 150.0)
    fwd = (src_box[0] * scale + px, src_box[1] * scale + py,
           src_box[2] * scale + px, src_box[3] * scale + py)
    back = map_back_letterbox(fwd, scale, px, py)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, src_box))
