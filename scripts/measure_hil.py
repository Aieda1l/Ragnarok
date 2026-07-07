"""Measure PC<->MCU round-trip latency over the Arduino serial link (box-only).

Flash firmware/ragnarok_mouse first, connect the 32u4 board, then:
  uv run python scripts/measure_hil.py COM5     (needs: pip install pyserial)
"""
from __future__ import annotations

import sys
import time

from ragnarok.aim.hil import measure_hil_roundtrip
from ragnarok.aim.protocol import START


def _read_frame(ser, timeout_s):
    ser.timeout = timeout_s
    b = ser.read(1)                        # sync to START
    if not b or b[0] != START:
        return None
    hdr = ser.read(3)                      # cmd + len16
    if len(hdr) < 3:
        return None
    length = hdr[1] | (hdr[2] << 8)
    payload = ser.read(length)
    crc = ser.read(1)
    if len(payload) < length or len(crc) < 1:
        return None
    return bytes([START]) + hdr + payload + crc


def main() -> None:
    import serial                          # lazy: pyserial only needed for this box-only tool
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    ser = serial.Serial(port, 115200, timeout=0.5)
    time.sleep(2.0)                         # 32u4 resets on serial open
    stats = measure_hil_roundtrip(ser.write, lambda to: _read_frame(ser, to), samples=30)
    ser.close()
    if stats is None:
        print(f"no echoes — is the board flashed + on {port}?")
        return
    print(f"HIL round-trip (ms): n={stats['n']}  min={stats['min_ms']:.2f}  "
          f"median={stats['median_ms']:.2f}  max={stats['max_ms']:.2f}")


if __name__ == "__main__":
    main()
