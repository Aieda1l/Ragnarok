from ragnarok.gui.dashboard_model import TelemetryHistory


class _Snap:
    def __init__(self, fps, p50, p99):
        self.fps, self.loop_ms_p50, self.loop_ms_p99 = fps, p50, p99


def test_history_records_series_and_stats():
    h = TelemetryHistory(maxlen=10)
    assert len(h) == 0 and h.series("fps") == ()
    h.push(fps=120.0, p50=5.0, p99=9.0)
    h.push_snapshot(_Snap(60.0, 8.0, 15.0))
    assert h.series("fps") == (120.0, 60.0)
    assert h.series("p50") == (5.0, 8.0) and h.series("p99") == (9.0, 15.0)
    assert h.stats() == {"fps": 60.0, "p50": 8.0, "p99": 15.0}     # latest values
    assert len(h) == 2


def test_history_is_bounded_ring_buffer():
    h = TelemetryHistory(maxlen=3)
    for i in range(5):
        h.push(fps=float(i), p50=0.0, p99=0.0)
    assert h.series("fps") == (2.0, 3.0, 4.0)                       # oldest dropped
    assert len(h) == 3


def test_empty_stats_is_zeroed():
    assert TelemetryHistory().stats() == {"fps": 0.0, "p50": 0.0, "p99": 0.0}


from ragnarok.gui.dashboard_model import sparkline_points


def test_sparkline_empty_and_single():
    assert sparkline_points([], x0=0, y0=0, w=100, h=50) == ()
    pts = sparkline_points([7.0], x0=0, y0=0, w=100, h=50)
    assert pts == ((0.0, 25.0),)                              # single -> mid-height


def test_sparkline_maps_endpoints_and_inverts_y():
    # increasing series over the full width; min at bottom, max at top
    pts = sparkline_points([0.0, 5.0, 10.0], x0=10, y0=20, w=100, h=40)
    assert pts[0] == (10.0, 60.0)                             # min -> bottom (y0+h)
    assert pts[-1] == (110.0, 20.0)                           # max -> top (y0)
    assert pts[1] == (60.0, 40.0)                             # mid x, mid y


def test_sparkline_flat_series_is_midline():
    pts = sparkline_points([3.0, 3.0, 3.0], x0=0, y0=0, w=90, h=30)
    assert [p[1] for p in pts] == [15.0, 15.0, 15.0]          # flat -> mid-line


def test_sparkline_explicit_range():
    # with y_min=0,y_max=100 a value of 50 sits at mid-height regardless of data
    pts = sparkline_points([50.0, 50.0], x0=0, y0=0, w=10, h=100, y_min=0.0, y_max=100.0)
    assert pts[0] == (0.0, 50.0)
