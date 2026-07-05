from ragnarok.core.types import Track, Team
from ragnarok.aim.head import resolve_aim_point
from ragnarok.aim.fov import aim_point


def _trk(tid, xyxy, cls):
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=cls, team=Team.ENEMY)


def test_head_and_body_modes_match_aim_point():
    body = _trk(1, (100.0, 100.0, 140.0, 260.0), cls=0)
    assert resolve_aim_point(body, (body,), mode="head", head_frac=0.15, head_class_id=1) \
        == aim_point(body, 0.15, "head")
    assert resolve_aim_point(body, (body,), mode="body", head_frac=0.15, head_class_id=1) \
        == aim_point(body, 0.15, "body")


def test_detected_head_aims_at_head_box_inside_body():
    body = _trk(1, (100.0, 100.0, 140.0, 260.0), cls=0)          # tall body
    head = _trk(2, (110.0, 105.0, 130.0, 130.0), cls=1)          # head box near top
    pt = resolve_aim_point(body, (body, head), mode="detected_head",
                           head_frac=0.15, head_class_id=1)
    assert pt == head.center                                     # aims at the detected head


def test_detected_head_falls_back_to_head_frac_when_no_head():
    body = _trk(1, (100.0, 100.0, 140.0, 260.0), cls=0)
    pt = resolve_aim_point(body, (body,), mode="detected_head",
                           head_frac=0.15, head_class_id=1)
    assert pt == aim_point(body, 0.15, "head")                   # fallback


def test_detected_head_when_target_is_itself_a_head():
    head = _trk(5, (110.0, 105.0, 130.0, 130.0), cls=1)
    pt = resolve_aim_point(head, (head,), mode="detected_head",
                           head_frac=0.15, head_class_id=1)
    assert pt == head.center


def test_head_outside_body_box_is_ignored():
    body = _trk(1, (100.0, 100.0, 140.0, 260.0), cls=0)
    far_head = _trk(2, (400.0, 400.0, 420.0, 420.0), cls=1)      # a different enemy's head
    pt = resolve_aim_point(body, (body, far_head), mode="detected_head",
                           head_frac=0.15, head_class_id=1)
    assert pt == aim_point(body, 0.15, "head")                   # not the far head
