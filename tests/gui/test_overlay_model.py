import math
from ragnarok.gui.overlay_model import ScreenMap, FovRing, OverlayScene


def test_screenmap_from_region_maps_points_and_rect():
    # ROI 384 captured at screen (100,50)-(484,434): scale 1.0, offset (100,50)
    m = ScreenMap.from_region((100, 50, 484, 434), 384, 384)
    assert m.pt(0, 0) == (100.0, 50.0)
    assert m.pt(192, 192) == (292.0, 242.0)          # ROI centre -> region centre
    assert m.rect((0, 0, 10, 20)) == (100.0, 50.0, 110.0, 70.0)


def test_screenmap_scales_when_region_larger_than_roi():
    m = ScreenMap.from_region((0, 0, 768, 768), 384, 384)  # 2x upscale
    assert m.scale_x == 2.0 and m.scale_y == 2.0
    assert m.pt(10, 10) == (20.0, 20.0)


def test_empty_scene_has_no_signal():
    s = OverlayScene.empty()
    assert s.has_signal is False and s.fov is None and s.markers == ()


def test_fovring_fields():
    r = FovRing(center=(5.0, 6.0), acquire_radius=10.0, retain_radius=20.0)
    assert r.center == (5.0, 6.0) and r.acquire_radius < r.retain_radius
    assert r.tick_count == 12
