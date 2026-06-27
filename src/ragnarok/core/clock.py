"""Monotonic, QPC-backed timing. perf_counter_ns is monotonic and NTP-immune;
only differences are meaningful (undefined epoch)."""
import time

def now_ns() -> int:
    return time.perf_counter_ns()

def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0
