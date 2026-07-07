"""Qt-free wall latency measurement orchestrator (box-only capture injected).

Oscillates the view via ``mouse`` while sampling the scene's optical flow via
``shift_fn``, then cross-correlates (``aim.latency.estimate_lag``) to the
round-trip latency. All I/O is injected so the loop is unit-testable with fakes;
the real run uses the worker's capturer + a SendInput driver + cv2 phase
correlation. Reused by both the worker measure-mode and scripts/measure_latency.py.
"""
from __future__ import annotations

import math
import time

import cv2

from ragnarok.aim.latency import estimate_lag
from ragnarok.recoil.wall_learner import measure_shift


class WallLatencyMeasurer:
    def __init__(self, capturer, mouse, *, duration_s: float = 2.5, amp: float = 40.0,
                 freq_hz: float = 3.0, shift_fn=None, clock=None) -> None:
        self._cap = capturer
        self._mouse = mouse
        self._dur = duration_s
        self._amp = amp
        self._freq = freq_hz
        self._shift = shift_fn or measure_shift
        self._clock = clock or time.perf_counter

    def run(self) -> float | None:
        prev_gray = None
        prev_pos = 0.0
        commanded, observed, times = [], [], []
        none_streak = 0
        t0 = self._clock()
        while self._clock() - t0 < self._dur:
            frame = self._cap.grab()
            if frame is None:
                none_streak += 1
                if none_streak > 300:          # capturer stopped/broken -> bail (no spin)
                    break
                continue
            none_streak = 0
            t = self._clock() - t0
            gray = self._to_gray(frame.image)
            if prev_gray is not None:
                dx, _ = self._shift(prev_gray, gray)
                observed.append(dx)
                times.append(t)
                pos = self._amp * math.sin(2.0 * math.pi * self._freq * t)
                commanded.append(pos - prev_pos)          # per-frame velocity of the sinusoid
                prev_pos = pos
                self._mouse.move_relative(commanded[-1], 0.0)
            prev_gray = gray
        n = min(len(commanded), len(observed))
        if n < 10 or len(times) < 2:
            return None
        dt = (times[-1] - times[0]) / (len(times) - 1)
        return estimate_lag(commanded[:n], observed[:n], dt, max_lag_frames=int(0.25 / dt))

    @staticmethod
    def _to_gray(img):
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
