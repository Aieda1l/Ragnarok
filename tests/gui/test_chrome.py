from ragnarok.gui.chrome import (
    corner_bracket_segments, accent_bar_rect, notch_polygon)


def test_corner_brackets_eight_segments_arms_inward():
    segs = corner_bracket_segments(0.0, 0.0, 100.0, 60.0, arm=10.0)
    assert len(segs) == 8                                    # 2 arms x 4 corners
    # top-left corner: horizontal arm goes +x, vertical arm goes +y
    assert segs[0] == ((0.0, 0.0), (10.0, 0.0))
    assert segs[1] == ((0.0, 0.0), (0.0, 10.0))
    # bottom-right corner: arms point inward (-x, -y)
    assert segs[6] == ((100.0, 60.0), (90.0, 60.0))
    assert segs[7] == ((100.0, 60.0), (100.0, 50.0))


def test_accent_bar_rect_is_left_edge():
    assert accent_bar_rect(5.0, 8.0, 105.0, 68.0, width=4.0) == (5.0, 8.0, 4.0, 60.0)


def test_notch_polygon_clips_top_right_corner():
    poly = notch_polygon(0.0, 0.0, 100.0, 50.0, cut=10.0)
    assert poly == ((0.0, 0.0), (90.0, 0.0), (100.0, 10.0), (100.0, 50.0), (0.0, 50.0))
    assert (100.0, 0.0) not in poly                         # the corner is cut off
