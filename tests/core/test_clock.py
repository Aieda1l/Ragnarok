from ragnarok.core.clock import now_ns, ns_to_ms

def test_now_ns_is_monotonic_int():
    a = now_ns()
    b = now_ns()
    assert isinstance(a, int) and isinstance(b, int)
    assert b >= a

def test_ns_to_ms():
    assert ns_to_ms(1_500_000) == 1.5
