# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from robustcov._native import (
    matrix_mahalanobis2_batch,
    native_available,
    weighted_tucker_scores_2d,
)


def test_matrix_mahalanobis_cpp_matches_python():
    if not native_available():
        pytest.skip("native extension unavailable")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(17, 5, 7))
    location = rng.normal(size=(5, 7))
    A = rng.normal(size=(5, 5))
    B = rng.normal(size=(7, 7))
    row_precision = A @ A.T + np.eye(5)
    column_precision = B @ B.T + np.eye(7)
    python = matrix_mahalanobis2_batch(
        X, location, row_precision, column_precision, backend="python"
    )
    cpp = matrix_mahalanobis2_batch(
        X, location, row_precision, column_precision, backend="cpp"
    )
    assert np.allclose(cpp, python, rtol=1e-12, atol=1e-11)


def test_weighted_tucker_scores_cpp_matches_python():
    if not native_available():
        pytest.skip("native extension unavailable")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(13, 6, 8))
    weights = rng.uniform(size=X.shape)
    weights[rng.random(X.shape) < 0.15] = 0.0
    center = rng.normal(size=(6, 8))
    U, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    python = weighted_tucker_scores_2d(
        X, weights, center, U, V, ridge=1e-7, backend="python"
    )
    cpp = weighted_tucker_scores_2d(
        X, weights, center, U, V, ridge=1e-7, backend="cpp"
    )
    assert np.allclose(cpp, python, rtol=1e-11, atol=1e-11)


def test_cpp_backend_request_errors_when_extension_missing(monkeypatch):
    import robustcov._native as native

    monkeypatch.setattr(native, "_cpp", None)
    with pytest.raises(RuntimeError, match="native extension"):
        native.resolve_backend("cpp")


def test_vector_mahalanobis_cpp_matches_python():
    if not native_available():
        pytest.skip("native extension unavailable")
    from robustcov._native import mahalanobis_squared_batch

    rng = np.random.default_rng(2)
    X = rng.normal(size=(503, 13))
    location = rng.normal(size=13)
    A = rng.normal(size=(13, 13))
    precision = A @ A.T + np.eye(13)
    python = mahalanobis_squared_batch(
        X, location, precision, backend="python"
    )
    cpp = mahalanobis_squared_batch(
        X, location, precision, backend="cpp"
    )
    assert np.allclose(cpp, python, rtol=1e-12, atol=1e-10)


def test_vector_mahalanobis_auto_matches_python_for_small_and_large_inputs():
    from robustcov._native import mahalanobis_squared_batch

    rng = np.random.default_rng(3)
    location = rng.normal(size=8)
    A = rng.normal(size=(8, 8))
    precision = A @ A.T + np.eye(8)
    for n in (7, 2000):
        X = rng.normal(size=(n, 8))
        expected = mahalanobis_squared_batch(
            X, location, precision, backend="python"
        )
        actual = mahalanobis_squared_batch(
            X, location, precision, backend="auto"
        )
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_matrix_mahalanobis_auto_keeps_medium_workloads_on_numpy(monkeypatch):
    import robustcov._native as native

    if not native_available():
        pytest.skip("native extension unavailable")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 5, 7))
    location = rng.normal(size=(5, 7))
    A = rng.normal(size=(5, 5))
    B = rng.normal(size=(7, 7))
    row_precision = A @ A.T + np.eye(5)
    column_precision = B @ B.T + np.eye(7)

    def unexpected_cpp_call(*args, **kwargs):
        raise AssertionError("auto backend should keep this workload on NumPy")

    monkeypatch.setattr(native._cpp, "matrix_mahalanobis2_batch", unexpected_cpp_call)
    result = native.matrix_mahalanobis2_batch(
        X, location, row_precision, column_precision, backend="auto"
    )
    assert result.shape == (100,)


def test_joint_diagonalization_cpp_matches_python():
    if not native_available():
        pytest.skip("native extension unavailable")
    import robustcov as rc

    rng = np.random.default_rng(5)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    diagonals = rng.normal(size=(16, 12))
    matrices = np.asarray(
        [basis @ np.diag(values) @ basis.T for values in diagonals]
    )
    python_rotation, python_diagonalized, python_info = (
        rc.joint_diagonalize_symmetric(matrices, backend="python")
    )
    cpp_rotation, cpp_diagonalized, cpp_info = rc.joint_diagonalize_symmetric(
        matrices, backend="cpp"
    )
    np.testing.assert_allclose(
        np.abs(cpp_rotation.T @ python_rotation), np.eye(12), atol=1e-8
    )
    np.testing.assert_allclose(
        np.sort(np.diagonal(cpp_diagonalized, axis1=1, axis2=2), axis=1),
        np.sort(np.diagonal(python_diagonalized, axis1=1, axis2=2), axis=1),
        rtol=1e-11,
        atol=1e-11,
    )
    assert cpp_info["off_diagonal_energy"] == pytest.approx(
        python_info["off_diagonal_energy"], rel=1e-8, abs=1e-20
    )
