"""Hardware-in-the-loop (HIL) latency diagnostic (spec §12).

Measures the PC<->MCU round-trip by sending DIAG frames and timing the firmware's
echo (which carries its own micros()). Pure/testable given injected ``send`` +
``read_frame``; the real serial/UDP read is box-only (scripts/measure_hil.py).
"""
from __future__ import annotations

import time

from ragnarok.aim.protocol import encode_diag, decode_frame, CMD_DIAG


def measure_hil_roundtrip(send, read_frame, *, samples: int = 20,
                          clock=time.perf_counter, timeout_s: float = 0.5):
    """Send ``samples`` DIAG frames, read the MCU echoes, return round-trip stats (ms).

    ``send(frame_bytes)`` writes a frame; ``read_frame(timeout_s)`` returns a full
    response frame (bytes) or None. Returns ``{n, min_ms, median_ms, max_ms}`` or
    None if no valid echoes arrived."""
    rtts: list[float] = []
    for seq in range(samples):
        t0 = clock()
        send(encode_diag(seq & 0xFF))
        frame = read_frame(timeout_s)
        if frame is None:
            continue
        try:
            cmd, _payload = decode_frame(frame)
        except ValueError:
            continue
        if cmd == CMD_DIAG:
            rtts.append((clock() - t0) * 1000.0)
    if not rtts:
        return None
    rtts.sort()
    return {"n": len(rtts), "min_ms": rtts[0],
            "median_ms": rtts[len(rtts) // 2], "max_ms": rtts[-1]}
