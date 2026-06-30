"""Tests for dynamic-ROI coordinate transforms (pure)."""
from __future__ import annotations
from ragnarok.detection.roi import letterbox_params, map_back_letterbox, crop_region_for, map_back_crop


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


def test_crop_centered_when_room():
    assert crop_region_for((500, 400), 192, 1920, 1080) == (404, 304, 192, 192)


def test_crop_clamped_at_top_left():
    assert crop_region_for((10, 10), 192, 1920, 1080) == (0, 0, 192, 192)


def test_crop_clamped_at_bottom_right():
    assert crop_region_for((1915, 1075), 192, 1920, 1080) == (1728, 888, 192, 192)


def test_crop_map_back_upscales_and_offsets():
    crop = (404, 304, 192, 192)                 # size 192 fed to a 384 engine -> r=0.5
    # an engine-space box at (0,0,384,384) maps to the full crop region
    back = map_back_crop((0.0, 0.0, 384.0, 384.0), crop, 384)
    assert back == (404.0, 304.0, 404.0 + 192.0, 304.0 + 192.0)


def test_crop_map_back_roundtrip():
    crop = (404, 304, 192, 192)
    full_box = (450.0, 350.0, 500.0, 420.0)     # known full-frame box inside the crop
    r = 192 / 384
    eng = ((full_box[0] - 404) / r, (full_box[1] - 304) / r,
           (full_box[2] - 404) / r, (full_box[3] - 304) / r)
    back = map_back_crop(eng, crop, 384)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, full_box))
