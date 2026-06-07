import numpy as np
import pytest

from robustcov.geometry import (
    affine_invariant_distance,
    det_normalize,
    logeuclidean_distance,
    spd_exp,
    spd_geodesic,
    spd_log,
    spd_power,
    trace_normalize,
    tyler_fixed_point_residual,
    tyler_objective,
)


def _spd():
    A = np.array([[2.0, 0.4], [0.4, 1.0]])
    return A


def test_trace_and_det_normalization():
    S = _spd()

    T = trace_normalize(S)
    assert np.allclose(np.trace(T), S.shape[0])

    D = det_normalize(S)
    assert np.allclose(np.linalg.det(D), 1.0)


def test_spd_log_exp_and_power():
    S = _spd()

    L = spd_log(S)
    S2 = spd_exp(L)
    assert np.allclose(S, S2)

    R = spd_power(S, 0.5)
    assert np.allclose(R @ R, S)

    I = spd_power(S, 0.0)
    assert np.allclose(I, np.eye(S.shape[0]))


def test_spd_distances_are_symmetric_and_zero_on_self():
    A = _spd()
    B = np.array([[1.4, -0.2], [-0.2, 2.5]])

    assert affine_invariant_distance(A, A) == pytest.approx(0.0, abs=1e-12)
    assert logeuclidean_distance(A, A) == pytest.approx(0.0, abs=1e-12)

    assert affine_invariant_distance(A, B) == pytest.approx(
        affine_invariant_distance(B, A)
    )
    assert logeuclidean_distance(A, B) == pytest.approx(logeuclidean_distance(B, A))


def test_spd_geodesic_endpoints():
    A = _spd()
    B = np.array([[1.4, -0.2], [-0.2, 2.5]])

    assert np.allclose(spd_geodesic(A, B, 0.0), A)
    assert np.allclose(spd_geodesic(A, B, 1.0), B)


def test_tyler_fixed_point_residual_zero_for_spherical_design():
    X = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    S = np.eye(2)

    resid = tyler_fixed_point_residual(S, X)
    assert resid == pytest.approx(0.0, abs=1e-12)


def test_tyler_objective_is_scale_invariant():
    X = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    S = np.array([[1.5, 0.2], [0.2, 0.8]])

    obj1 = tyler_objective(S, X, normalize=None)
    obj2 = tyler_objective(4.0 * S, X, normalize=None)

    assert obj1 == pytest.approx(obj2)
