# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Transformation and singular-data guarantees for structured estimators."""

from __future__ import annotations

import numpy as np
import pytest

import robustcov as rc


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(actual - expected) / denominator)


def _fast_mmcd(**kwargs):
    defaults = dict(
        support_fraction=0.8,
        quality="fast",
        n_init=10,
        n_best=3,
        initial_c_steps=2,
        max_iter=12,
        flip_flop_max_iter=25,
        reweight=False,
        random_state=3,
        backend="python",
    )
    defaults.update(kwargs)
    return rc.MMCD(**defaults)


def _fast_kmrcd(**kwargs):
    defaults = dict(
        support_fraction=0.8,
        regularization=0.2,
        n_init=10,
        n_best=3,
        initial_c_steps=2,
        max_iter=20,
        random_state=2,
    )
    defaults.update(kwargs)
    return rc.KMRCD(**defaults)


class _EmpiricalScatter:
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.location_ = X.mean(axis=0)
        self.covariance_ = np.cov(X, rowvar=False, ddof=1)
        return self


def _matrix_low_rank_sample(seed=0, n=48, rows=5, columns=6, ranks=(2, 2)):
    rng = np.random.default_rng(seed)
    row_components, _ = np.linalg.qr(rng.normal(size=(rows, ranks[0])))
    column_components, _ = np.linalg.qr(rng.normal(size=(columns, ranks[1])))
    cores = rng.normal(size=(n, *ranks))
    X = np.einsum(
        "au,nuv,bv->nab",
        row_components,
        cores,
        column_components,
        optimize=True,
    )
    X += 0.05 * rng.normal(size=X.shape)
    return rng, X


def test_mmcd_is_scale_and_orthogonally_equivariant_with_relative_ridge():
    rng = np.random.default_rng(701)
    X = rng.normal(size=(64, 3, 4))
    row_rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    column_rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    shift = rng.normal(size=(3, 4))
    scale = 1e-50

    reference = _fast_mmcd().fit(X)
    tiny = _fast_mmcd().fit(scale * X)
    transformed_data = np.asarray(
        [2.5 * row_rotation @ matrix @ column_rotation.T + shift for matrix in X]
    )
    transformed = _fast_mmcd().fit(transformed_data)

    np.testing.assert_array_equal(reference.raw_support_, tiny.raw_support_)
    np.testing.assert_allclose(tiny.location_ / scale, reference.location_, rtol=2e-10)
    np.testing.assert_allclose(
        tiny.kronecker_covariance() / scale**2,
        reference.kronecker_covariance(),
        rtol=2e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(tiny.distances_, reference.distances_, rtol=2e-8)

    np.testing.assert_array_equal(reference.raw_support_, transformed.raw_support_)
    expected_location = 2.5 * row_rotation @ reference.location_ @ column_rotation.T + shift
    np.testing.assert_allclose(transformed.location_, expected_location, rtol=2e-8, atol=2e-9)
    transform = np.kron(column_rotation, row_rotation)
    expected_covariance = (
        2.5**2
        * transform
        @ reference.kronecker_covariance()
        @ transform.T
    )
    np.testing.assert_allclose(
        transformed.kronecker_covariance(), expected_covariance, rtol=3e-8, atol=2e-9
    )
    np.testing.assert_allclose(transformed.distances_, reference.distances_, rtol=3e-8)


@pytest.mark.filterwarnings(
    "ignore:MMCD subset polishing reached the iteration limit.*"
)
def test_mmcd_regularizes_rank_deficient_samples_and_zero_ridge_fails_clearly():
    rng = np.random.default_rng(702)
    row_direction = np.array([1.0, 2.0, 3.0])
    column_direction = np.array([1.0, -0.5, 2.0, 0.25])
    amplitudes = rng.normal(size=42)
    X = (
        amplitudes[:, None, None]
        * row_direction[None, :, None]
        * column_direction[None, None, :]
    )

    fitted = _fast_mmcd(ridge=1e-8, max_iter=6, flip_flop_max_iter=12).fit(X)
    assert np.linalg.eigvalsh(fitted.row_covariance_).min() > 0.0
    assert np.linalg.eigvalsh(fitted.column_covariance_).min() > 0.0
    assert np.isfinite(fitted.distances_).all()

    with pytest.raises(RuntimeError, match="increase ridge"):
        _fast_mmcd(ridge=0.0, max_iter=4, flip_flop_max_iter=8).fit(X)


def test_kmrcd_rbf_median_bandwidth_is_invariant_to_raw_measurement_scale():
    rng = np.random.default_rng(703)
    X = rng.normal(size=(68, 4))
    scale = 1e-50
    kwargs = dict(kernel="rbf", standardization="none", gamma="median")

    reference = _fast_kmrcd(**kwargs).fit(X)
    tiny = _fast_kmrcd(**kwargs).fit(scale * X)

    assert tiny.gamma_ * scale**2 == pytest.approx(reference.gamma_, rel=2e-14)
    np.testing.assert_array_equal(tiny.support_, reference.support_)
    np.testing.assert_allclose(tiny.kernel_matrix_, reference.kernel_matrix_, rtol=2e-14)
    np.testing.assert_allclose(tiny.distances_, reference.distances_, rtol=2e-12)


def test_kmrcd_uses_relative_psd_validation_and_handles_duplicate_rows():
    materially_indefinite = 1e-10 * np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        _fast_kmrcd(
            kernel="precomputed",
            support_fraction=1.0,
            n_init=1,
            n_best=1,
        ).fit(materially_indefinite)

    rng = np.random.default_rng(704)
    X = np.repeat(rng.normal(size=(12, 3)), 2, axis=0)
    fitted = _fast_kmrcd(kernel="linear", regularization="auto").fit(X)
    assert np.linalg.eigvalsh(fitted.regularized_kernel_).min() > 0.0
    assert np.isfinite(fitted.distances_).all()


def test_cellmcd_is_featurewise_affine_and_permutation_equivariant():
    rng = np.random.default_rng(705)
    X = rng.normal(size=(78, 5))
    X[:5, 0] += 7.0
    X[2, 3] = np.nan
    scales = np.array([1e-50, 2.0, 0.2, 7.0, 1e20])
    offsets = np.array([1e-50, 3.0, -2.0, 9.0, -1e20])
    permutation = np.array([4, 1, 3, 0, 2])
    inverse = np.argsort(permutation)
    kwargs = dict(alpha=0.8, max_iter=35, min_samples_per_feature=None)

    reference = rc.CellMCD(**kwargs).fit(X)
    transformed = rc.CellMCD(**kwargs).fit(X * scales + offsets)
    permuted = rc.CellMCD(**kwargs).fit(X[:, permutation])

    np.testing.assert_array_equal(reference.cell_support_, transformed.cell_support_)
    np.testing.assert_allclose(
        transformed.location_, reference.location_ * scales + offsets, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        transformed.covariance_,
        reference.covariance_ * np.outer(scales, scales),
        rtol=3e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        transformed.standardized_residuals_, reference.standardized_residuals_, rtol=2e-11
    )

    np.testing.assert_array_equal(reference.cell_support_, permuted.cell_support_[:, inverse])
    np.testing.assert_allclose(reference.location_, permuted.location_[inverse], rtol=2e-12)
    np.testing.assert_allclose(
        reference.covariance_, permuted.covariance_[np.ix_(inverse, inverse)], rtol=3e-12
    )


def test_cellmcd_rejects_singular_constant_features_clearly():
    rng = np.random.default_rng(706)
    X = rng.normal(size=(64, 4))
    X[:, 2] = 3.0

    with pytest.raises(ValueError, match="nonzero robust scale.*2"):
        rc.CellMCD(max_iter=25, min_samples_per_feature=None).fit(X)
    with pytest.raises(ValueError, match="nonzero robust scale"):
        rc.CellMCD(min_samples_per_feature=None).fit(np.ones((30, 4)))


def test_robust_pca_preserves_tiny_units_and_orthogonal_subspaces():
    rng = np.random.default_rng(707)
    X = rng.normal(size=(130, 6)) @ np.diag([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    scale = 1e-50
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))

    reference = rc.RobustPCA(
        n_components=3, estimator=_EmpiricalScatter(), whiten=True
    ).fit(X)
    tiny = rc.RobustPCA(
        n_components=3, estimator=_EmpiricalScatter(), whiten=True
    ).fit(scale * X)
    rotated = rc.RobustPCA(
        n_components=3, estimator=_EmpiricalScatter(), whiten=True
    ).fit(X @ rotation)

    np.testing.assert_allclose(tiny.eigenvalues_ / scale**2, reference.eigenvalues_, rtol=2e-13)
    np.testing.assert_allclose(
        tiny.components_.T @ tiny.components_,
        reference.components_.T @ reference.components_,
        rtol=2e-13,
        atol=2e-13,
    )
    expected_projection = (
        rotation.T
        @ (reference.components_.T @ reference.components_)
        @ rotation
    )
    np.testing.assert_allclose(
        rotated.components_.T @ rotated.components_, expected_projection, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(rotated.eigenvalues_, reference.eigenvalues_, rtol=2e-12)
    np.testing.assert_allclose(
        np.cov(reference.transform(X), rowvar=False), np.eye(3), rtol=2e-12, atol=2e-12
    )


def test_robust_pca_regularizes_actual_singular_data():
    rng = np.random.default_rng(708)
    base = rng.normal(size=(80, 3))
    X = np.column_stack([base, base[:, 0], np.ones(base.shape[0])])

    fitted = rc.RobustPCA(estimator=_EmpiricalScatter(), ridge=1e-8).fit(X)
    assert np.linalg.eigvalsh(fitted.covariance_).min() > 0.0
    assert fitted.eigenvalue_floor_ > 0.0
    assert np.isfinite(fitted.transform(X)).all()


def test_multilinear_pca_is_scale_and_mode_permutation_equivariant():
    rng, X = _matrix_low_rank_sample(seed=709)
    X[0, 1, 2] = np.nan
    scale = 1e-50
    row_permutation = np.array([4, 0, 3, 1, 2])
    column_permutation = np.array([3, 5, 1, 0, 4, 2])
    inverse_rows = np.argsort(row_permutation)
    inverse_columns = np.argsort(column_permutation)
    kwargs = dict(ranks=(2, 2), max_iter=22, backend="python")

    reference = rc.RobustMultilinearPCA(**kwargs).fit(X)
    tiny = rc.RobustMultilinearPCA(**kwargs).fit(scale * X)
    permuted = rc.RobustMultilinearPCA(**kwargs).fit(
        X[:, row_permutation][:, :, column_permutation]
    )

    np.testing.assert_allclose(tiny.fitted_values_ / scale, reference.fitted_values_, rtol=3e-11)
    np.testing.assert_allclose(tiny.residual_scales_ / scale, reference.residual_scales_, rtol=3e-11)
    np.testing.assert_array_equal(tiny.cell_outlier_mask_, reference.cell_outlier_mask_)
    np.testing.assert_array_equal(tiny.case_outlier_mask_, reference.case_outlier_mask_)

    restored_fitted = permuted.fitted_values_[:, inverse_rows][:, :, inverse_columns]
    np.testing.assert_allclose(restored_fitted, reference.fitted_values_, rtol=3e-11, atol=3e-12)
    restored_row_projection = (permuted.row_components_ @ permuted.row_components_.T)[
        np.ix_(inverse_rows, inverse_rows)
    ]
    restored_column_projection = (
        permuted.column_components_ @ permuted.column_components_.T
    )[np.ix_(inverse_columns, inverse_columns)]
    np.testing.assert_allclose(
        restored_row_projection,
        reference.row_components_ @ reference.row_components_.T,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        restored_column_projection,
        reference.column_components_ @ reference.column_components_.T,
        rtol=3e-11,
        atol=3e-12,
    )
    np.testing.assert_array_equal(
        permuted.cell_outlier_mask_[:, inverse_rows][:, :, inverse_columns],
        reference.cell_outlier_mask_,
    )


def test_multilinear_pca_rejects_completely_constant_samples():
    with pytest.raises(ValueError, match="nonzero variation"):
        rc.RobustMultilinearPCA(ranks=(1, 1), max_iter=5).fit(np.ones((12, 3, 3)))
