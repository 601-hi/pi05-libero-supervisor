import numpy as np

from train_natural_multiresponse_dual_expert import quaternion_delta_rotvec, robust_fit


def test_identity_quaternion_delta_is_zero():
    np.testing.assert_allclose(quaternion_delta_rotvec([0, 0, 0, 1], [0, 0, 0, 1]), 0, atol=1e-12)


def test_quaternion_sign_does_not_change_rotation():
    q = np.asarray([0.1, -0.2, 0.3, 0.92736185]); q /= np.linalg.norm(q)
    np.testing.assert_allclose(quaternion_delta_rotvec(-q, q), 0, atol=1e-7)


def test_robust_scale_is_positive_for_constant_columns():
    _, scale = robust_fit(np.ones((5, 3)))
    assert np.all(scale > 0)
