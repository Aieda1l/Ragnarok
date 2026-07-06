from ragnarok.core.types import Track, Tracks, Team
from ragnarok.aim.select import select_target


def _t(tid, xyxy, cls):
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=cls, team=Team.ENEMY)


def test_head_inside_body_is_not_a_separate_candidate():
    body = _t(1, (90.0, 90.0, 110.0, 260.0), 0)     # body
    head = _t(2, (95.0, 95.0, 105.0, 115.0), 1)     # head enclosed by the body
    chosen = select_target(Tracks(items=(body, head)), (100.0, 105.0),
                           fov_px=200.0, head_class_id=1)
    assert chosen == 1                              # body wins; head only refines it


def test_standalone_head_is_still_selectable():
    head = _t(5, (95.0, 95.0, 105.0, 115.0), 1)     # head with no enclosing body
    chosen = select_target(Tracks(items=(head,)), (100.0, 105.0),
                           fov_px=200.0, head_class_id=1)
    assert chosen == 5                              # head-only detection still engages


def test_without_head_class_id_behavior_unchanged():
    body = _t(1, (90.0, 90.0, 110.0, 260.0), 0)
    head = _t(2, (95.0, 95.0, 105.0, 115.0), 1)
    chosen = select_target(Tracks(items=(body, head)), (100.0, 105.0), fov_px=200.0)
    assert chosen in (1, 2)                         # feature off -> both compete (old path)
