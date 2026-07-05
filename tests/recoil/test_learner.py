from ragnarok.recoil.learner import estimate_recoil_pattern, average_patterns


def test_estimate_cumsums_per_shot_kicks():
    # per-shot drift (0,2),(0,3),(1,2) -> cumulative (0,2),(0,5),(1,7)
    assert estimate_recoil_pattern([(0, 2), (0, 3), (1, 2)]) == ((0.0, 2.0), (0.0, 5.0), (1.0, 7.0))
    assert estimate_recoil_pattern([]) == ()


def test_average_patterns_elementwise_mean_truncated():
    a = ((0.0, 2.0), (0.0, 4.0), (0.0, 6.0))
    b = ((0.0, 4.0), (0.0, 6.0))                     # shorter -> truncate to len 2
    assert average_patterns([a, b]) == ((0.0, 3.0), (0.0, 5.0))
    assert average_patterns([]) == ()
    assert average_patterns([(), a]) == a            # empty patterns ignored
