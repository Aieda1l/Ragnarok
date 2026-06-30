"""Tests for the DynamicRoiPlanner (FSM + transforms tied together)."""
from __future__ import annotations
from ragnarok.config.schema import DynamicRoiConfig
from ragnarok.detection.roi import DynamicRoiPlanner, RoiMode


def test_search_plan_is_full_frame_letterboxed():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384))
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=None,
                  frame_index=0, has_lock=False)
    assert plan.mode == RoiMode.SEARCH
    assert plan.region == (0, 0, 1920, 1080) and plan.letterboxed is True


def test_track_plan_is_crop_after_lock():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384,
                                           rescan_interval_frames=0))
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=0, has_lock=True)                    # -> TRACK
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                  frame_index=1, has_lock=True)
    assert plan.mode == RoiMode.TRACK
    assert plan.region == (404, 304, 192, 192) and plan.letterboxed is False


def test_rescan_forces_full_frame_while_tracking():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, rescan_interval_frames=5))
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=1, has_lock=True)                    # TRACK
    plan = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                  frame_index=5, has_lock=True)             # 5 % 5 == 0 -> rescan
    assert plan.mode == RoiMode.TRACK                       # logical mode unchanged
    assert plan.region == (0, 0, 1920, 1080) and plan.letterboxed is True


def test_map_back_search_then_track():
    p = DynamicRoiPlanner(DynamicRoiConfig(track_roi_size=192, model_input_px=384,
                                           rescan_interval_frames=0))
    # SEARCH: full 1920x1080 letterboxed into 384
    sp = p.plan(frame_w=1920, frame_h=1080, target_center=None,
                frame_index=0, has_lock=False)
    full = p.map_back((192.0, 0.0, 384.0, 216.0), sp)       # engine-space box
    # 1920x1080 -> scale 0.2, pad_y (384-216)/2=84; (192,0)->( (192)/0.2, (0-84)/0.2 ),
    # (384,216)->( 384/0.2, (216-84)/0.2 ) = (960,-420,1920,660) (+region origin 0,0)
    assert full == (960.0, -420.0, 1920.0, 660.0)
    # TRACK crop map-back lands inside the crop region
    p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
           frame_index=1, has_lock=True)
    tp = p.plan(frame_w=1920, frame_h=1080, target_center=(500, 400),
                frame_index=2, has_lock=True)
    tback = p.map_back((0.0, 0.0, 384.0, 384.0), tp)
    assert tback == (404.0, 304.0, 404.0 + 192.0, 304.0 + 192.0)
