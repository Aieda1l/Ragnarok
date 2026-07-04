import math
from ragnarok.gui.overlay_model import ScreenMap, FovBox, OverlayScene


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
    assert s.fov_thin == () and s.fov_thick == ()


def test_fovbox_fields():
    b = FovBox(center=(960.0, 540.0), half=40.0)
    assert b.center == (960.0, 540.0) and b.half == 40.0


from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay_model import build_markers, fov_bracket_segments


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


def test_fov_bracket_segments_two_verticals_and_four_diagonal_arms():
    thin, thick = fov_bracket_segments((100.0, 100.0), half=40.0, arm=10.0)
    # two thin verticals at x = 60 and x = 140, spanning the full square height
    assert thin == (((60.0, 60.0), (60.0, 140.0)),
                    ((140.0, 60.0), (140.0, 140.0)))
    # four bold arms, each a true 45° diagonal (|dx| == |dy| == arm) flaring OUT
    assert len(thick) == 4
    for (ax, ay), (bx, by) in thick:
        assert abs(bx - ax) == 10.0 and abs(by - ay) == 10.0        # 45°, length arm
    # left-top arm starts at the top-left corner, heads UP-and-LEFT (up + outward)
    assert thick[0] == ((60.0, 60.0), (50.0, 50.0))
    # right-bottom arm starts at bottom-right, heads DOWN-and-RIGHT (down + outward)
    assert thick[3] == ((140.0, 140.0), (150.0, 150.0))


def test_ray_rect_edge_and_in_viewport():
    from ragnarok.gui.overlay_model import _ray_rect_edge, _in_viewport
    vp = (0.0, 0.0, 100.0, 100.0)
    assert _in_viewport((50.0, 50.0), vp) is True
    assert _in_viewport((150.0, 50.0), vp) is False
    # ray from centre toward a point far to the right hits the x=100 edge at y=50
    edge = _ray_rect_edge((50.0, 50.0), (500.0, 50.0), vp)
    assert edge == (100.0, 50.0)
    # toward bottom-right corner hits the corner
    edge2 = _ray_rect_edge((50.0, 50.0), (150.0, 150.0), vp)
    assert edge2 == (100.0, 100.0)


from ragnarok.config.schema import AppConfig
from ragnarok.telemetry.snapshot import TelemetrySnapshot
from ragnarok.gui.overlay_model import build_scene


def _snap(**kw):
    base = dict(fps=1.0, loop_ms_p50=1.0, loop_ms_p99=1.0, detection_count=0,
                preview=None, seq=1)
    base.update(kw)
    return TelemetrySnapshot(**base)


def test_build_scene_no_region_is_empty():
    scene = build_scene(snapshot=_snap(roi_region=None), cfg=AppConfig(),
                        viewport=(0.0, 0.0, 1920.0, 1080.0))
    assert scene.has_signal is False


def test_build_scene_has_square_fov_brackets_and_locked_line():
    cfg = AppConfig()                                   # roi 384, hfov 90, screen 1920
    # ROI centred on screen: region (768,348)-(1152,732); crosshair -> (960,540)
    tracks = (_trk(3, (180, 150, 204, 234)),)           # enemy near crosshair
    snap = _snap(roi_region=(768, 348, 1152, 732), tracks=tracks, locked_target_id=3)
    scene = build_scene(snapshot=snap, cfg=cfg, viewport=(0.0, 0.0, 1920.0, 1080.0))
    assert scene.has_signal is True
    assert scene.crosshair == (960.0, 540.0)
    assert isinstance(scene.fov, FovBox) and scene.fov.center == (960.0, 540.0)
    assert scene.fov.half > 0.0
    assert len(scene.fov_thin) == 2 and len(scene.fov_thick) == 4   # two brackets
    assert scene.locked_line is not None
    assert scene.locked_line[0] == scene.crosshair                  # from crosshair
    assert len(scene.markers) == 1


def test_build_scene_no_lock_has_no_line():
    cfg = AppConfig()
    tracks = (_trk(3, (180, 150, 204, 234)),)
    snap = _snap(roi_region=(768, 348, 1152, 732), tracks=tracks, locked_target_id=None)
    scene = build_scene(snapshot=snap, cfg=cfg, viewport=(0.0, 0.0, 1920.0, 1080.0))
    assert scene.locked_line is None and len(scene.fov_thick) == 4


def test_build_scene_offscreen_enemy_becomes_hint():
    cfg = AppConfig()
    # enemy far outside the ROI mapping -> diamond outside a tiny viewport
    tracks = (_trk(9, (380, 380, 384, 384)),)
    snap = _snap(roi_region=(768, 348, 1152, 732), tracks=tracks, locked_target_id=None)
    scene = build_scene(snapshot=snap, cfg=cfg, viewport=(0.0, 0.0, 1000.0, 700.0))
    assert len(scene.offscreen) == 1
    assert scene.offscreen[0].team == Team.ENEMY
