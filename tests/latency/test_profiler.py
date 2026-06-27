from ragnarok.latency.profiler import StageProfiler

def test_percentiles_basic():
    p = StageProfiler(window=100)
    for i in range(100):
        p.record("infer", (i + 1) * 1_000_000)  # 1ms..100ms
    p50, p99 = p.percentiles("infer")
    assert 49.0 <= p50 <= 52.0
    assert p99 >= 98.0

def test_unknown_stage_returns_zeros():
    p = StageProfiler()
    assert p.percentiles("nope") == (0.0, 0.0)

def test_window_evicts_old():
    p = StageProfiler(window=3)
    for v in [10, 20, 30, 40]:  # ms
        p.record("s", v * 1_000_000)
    p50, _ = p.percentiles("s")
    assert p50 == 30.0  # only [20,30,40] retained
