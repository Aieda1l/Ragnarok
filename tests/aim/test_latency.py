import numpy as np

from ragnarok.aim.latency import estimate_lag


def test_recovers_known_lag():
    rng = np.random.default_rng(0)
    commanded = rng.standard_normal(200)
    lag = 5
    # scene flow = -commanded, delayed by `lag` frames (observed[i] = -commanded[i-lag])
    observed = np.zeros(200)
    observed[lag:] = -commanded[:-lag]
    est = estimate_lag(commanded, observed, dt_s=0.01, max_lag_frames=20)
    assert abs(est - lag * 0.01) < 1e-9              # 0.05 s


def test_zero_lag():
    rng = np.random.default_rng(1)
    commanded = rng.standard_normal(100)
    observed = -commanded                            # instant response
    assert estimate_lag(commanded, observed, dt_s=0.01, max_lag_frames=20) == 0.0


def test_guards_insufficient_signal():
    assert estimate_lag([1.0, 2.0], [0.0, 0.0], 0.01, 5) is None      # too short
    assert estimate_lag([0.0] * 10, [0.0] * 10, 0.01, 5) is None      # flat -> no corr
    assert estimate_lag([1.0] * 10, [1.0] * 8, 0.01, 5) is None       # length mismatch
