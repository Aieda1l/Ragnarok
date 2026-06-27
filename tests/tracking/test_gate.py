import numpy as np

from ragnarok.tracking.gate import CHI2_GATE_2DOF, mahalanobis_gate


def test_chi2_constant():
    assert abs(CHI2_GATE_2DOF - 5.9915) < 1e-6


def test_near_det_kept_far_det_gated():
    # one track at origin with isotropic covariance sigma^2 = 1
    track_mean_xy = np.array([[0.0, 0.0]])
    track_S_xy = np.array([np.eye(2)])
    # det 0 is at d^2 = 2 (< 5.9915 -> kept); det 1 is at d^2 = 50 (> gate -> +inf)
    det_xy = np.array([[1.0, 1.0], [5.0, 5.0]])
    cost = np.zeros((1, 2))
    out = mahalanobis_gate(cost, track_mean_xy, track_S_xy, det_xy)
    assert np.isfinite(out[0, 0])
    assert out[0, 0] == 0.0
    assert np.isinf(out[0, 1])


def test_anisotropic_covariance():
    track_mean_xy = np.array([[10.0, 10.0]])
    track_S_xy = np.array([np.diag([4.0, 1.0])])  # sigma_x^2=4, sigma_y^2=1
    # det at (14,10): d^2 = 16/4 = 4 < gate ; det at (16,10): d^2 = 36/4 = 9 > gate
    det_xy = np.array([[14.0, 10.0], [16.0, 10.0]])
    cost = np.zeros((1, 2))
    out = mahalanobis_gate(cost, track_mean_xy, track_S_xy, det_xy)
    assert np.isfinite(out[0, 0])
    assert np.isinf(out[0, 1])
