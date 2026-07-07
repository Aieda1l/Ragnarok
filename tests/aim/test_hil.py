import struct

from ragnarok.aim.hil import measure_hil_roundtrip
from ragnarok.aim.protocol import _frame, CMD_DIAG


def _echo_frame():
    return _frame(CMD_DIAG, struct.pack("<I", 100))    # firmware micros() echo


def test_roundtrip_stats_from_echoes():
    sent = []
    t = {"s": 0.0}

    def clock():
        v = t["s"]
        t["s"] += 0.001                                # each call +1 ms -> 1 ms RTT/sample
        return v

    stats = measure_hil_roundtrip(sent.append, lambda to: _echo_frame(),
                                  samples=5, clock=clock)
    assert stats["n"] == 5
    assert abs(stats["median_ms"] - 1.0) < 1e-9
    assert len(sent) == 5                              # one DIAG frame per sample


def test_no_echo_returns_none():
    assert measure_hil_roundtrip(lambda b: None, lambda to: None, samples=3) is None


def test_garbage_frame_is_skipped():
    assert measure_hil_roundtrip(lambda b: None, lambda to: b"\x00\x00", samples=3) is None
