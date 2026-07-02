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


from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay_model import build_markers, LockAgeTracker


def _trk(tid, xyxy, team=Team.ENEMY, conf=0.9):
    return Track(track_id=tid, xyxy=xyxy, confidence=conf, class_id=0, team=team)


def test_build_markers_sets_lock_diamond_and_fov():
    m = ScreenMap.from_region((0, 0, 384, 384), 384, 384)  # identity
    crosshair = (192.0, 192.0)
    tracks = (
        _trk(1, (180, 150, 204, 234)),   # near crosshair -> in_fov, will be locked
        _trk(2, (10, 10, 30, 90)),       # far corner -> out of fov
    )
    markers = build_markers(tracks, m, crosshair, fov_px=40.0, locked_id=1,
                            head_frac=0.15, aim_mode="head")
    by_id = {mk.track_id: mk for mk in markers}
    assert by_id[1].locked is True and by_id[2].locked is False
    assert by_id[1].in_fov is True and by_id[2].in_fov is False
    # diamond is the head aim-point: x = box centre, y = y1 + 0.15*height
    assert by_id[1].diamond == (192.0, 150.0 + 0.15 * 84.0)
    assert by_id[1].box == (180.0, 150.0, 204.0, 234.0)


def test_lock_age_tracker_resets_on_change():
    t = LockAgeTracker()
    assert t.update(None, 0) == 0.0
    assert t.update(5, 1_000_000_000) == 0.0        # first frame of a new lock -> age 0
    assert t.update(5, 1_500_000_000) == 0.5        # 0.5 s later
    assert t.update(6, 1_600_000_000) == 0.0        # switched lock -> reset
    assert t.update(None, 2_000_000_000) == 0.0     # lock dropped -> 0


from ragnarok.gui.overlay_model import (
    lock_progress, bracket_segments, _ray_rect_edge, _in_viewport)


def test_lock_progress_clamps():
    assert lock_progress(-1.0, 0.2) == 0.0
    assert lock_progress(0.1, 0.2) == 0.5
    assert lock_progress(5.0, 0.2) == 1.0
    assert lock_progress(0.0, 0.0) == 1.0            # zero-duration -> snapped


def test_bracket_segments_converge():
    box = (100.0, 100.0, 200.0, 200.0)
    # t=1: top-left corner arm starts exactly at (100,100)
    snapped = bracket_segments(box, t=1.0, gap=20.0, arm_len=10.0)
    tl_h = snapped[0]                                # first corner, horizontal arm
    assert tl_h[0] == (100.0, 100.0)
    assert tl_h[1] == (110.0, 100.0)                 # arm extends inward +x
    # t=0: same corner is offset gap px up-and-left (unconverged/wide)
    wide = bracket_segments(box, t=0.0, gap=20.0, arm_len=10.0)
    assert wide[0][0] == (80.0, 80.0)
    assert len(snapped) == 8                          # 4 corners x 2 arms


def test_ray_rect_edge_and_in_viewport():
    vp = (0.0, 0.0, 100.0, 100.0)
    assert _in_viewport((50.0, 50.0), vp) is True
    assert _in_viewport((150.0, 50.0), vp) is False
    # ray from centre toward a point far to the right hits the x=100 edge at y=50
    edge = _ray_rect_edge((50.0, 50.0), (500.0, 50.0), vp)
    assert edge == (100.0, 50.0)
    # toward bottom-right corner hits the corner
    edge2 = _ray_rect_edge((50.0, 50.0), (150.0, 150.0), vp)
    assert edge2 == (100.0, 100.0)
