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
