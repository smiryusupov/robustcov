from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

import robustcov as rc


def _mixing_matrix():
    return np.array(
        [[1.0, 0.4, -0.2], [0.2, 1.2, 0.5], [-0.4, 0.3, 0.9]],
        dtype=float,
    )


def _independent_sample(seed=0, n_samples=3000):
    rng = np.random.default_rng(seed)
    sources = np.column_stack(
        [
            rng.laplace(size=n_samples) / np.sqrt(2.0),
            rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples),
            rng.normal(size=n_samples),
        ]
    )
    sources -= np.mean(sources, axis=0)
    sources /= np.std(sources, axis=0)
    mixing = _mixing_matrix()
    return sources @ mixing.T, sources, mixing


def _ar_sources(seed=0, n_samples=2500):
    rng = np.random.default_rng(seed)
    coefficients = np.array([0.85, -0.55, 0.20])
    innovations = rng.normal(size=(n_samples, coefficients.size))
    sources = np.zeros_like(innovations)
    for index in range(1, n_samples):
        sources[index] = coefficients * sources[index - 1] + innovations[index]
    sources -= np.mean(sources, axis=0)
    sources /= np.std(sources, axis=0)
    mixing = _mixing_matrix()
    return sources @ mixing.T, sources, mixing


def test_joint_diagonalization_recovers_common_eigenvectors():
    rng = np.random.default_rng(12)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    diagonals = rng.normal(size=(8, 5))
    matrices = np.asarray(
        [rotation @ np.diag(values) @ rotation.T for values in diagonals]
    )

    fitted_rotation, transformed, info = rc.joint_diagonalize_symmetric(matrices)

    assert info["converged"]
    assert info["off_diagonal_energy"] < 1e-16
    np.testing.assert_allclose(fitted_rotation.T @ fitted_rotation, np.eye(5), atol=1e-12)
    alignment = np.abs(rotation.T @ fitted_rotation)
    assert np.all(np.max(alignment, axis=1) > 1.0 - 1e-9)
    assert rc.off_diagonal_energy(transformed) < 1e-16


def test_bss_indices_ignore_permutation_sign_and_scale():
    mixing = _mixing_matrix()
    exact = np.linalg.inv(mixing)
    permutation = np.array([2, 0, 1])
    scaled = exact[permutation] * np.array([-2.0, 0.5, 3.0])[:, None]
    assert rc.minimum_distance_index(scaled, mixing) < 1e-12
    assert rc.amari_index(scaled, mixing) < 1e-12


def test_two_scatter_ica_recovers_sources_and_is_unit_equivariant():
    X, _, mixing = _independent_sample(seed=3)
    estimator = rc.TwoScatterICA(
        radial_clip_quantile=0.90,
        max_pairs=15000,
        random_state=0,
    ).fit(X)
    assert rc.minimum_distance_index(estimator.unmixing_, mixing) < 0.12
    assert estimator.sources_.shape == X.shape
    np.testing.assert_allclose(
        estimator.inverse_transform(estimator.transform(X)), X, rtol=1e-9, atol=1e-9
    )

    feature_scale = np.array([1e-30, 3e-30, 7e-30])
    scaled = rc.TwoScatterICA(
        radial_clip_quantile=0.90,
        max_pairs=15000,
        random_state=0,
    ).fit(X * feature_scale)
    scaled_mixing = np.diag(feature_scale) @ mixing
    assert rc.minimum_distance_index(scaled.unmixing_, scaled_mixing) < 0.12


def test_two_scatter_ica_bounds_gross_outlier_contributions():
    X, _, mixing = _independent_sample(seed=4)
    rng = np.random.default_rng(44)
    contaminated = X.copy()
    rows = rng.choice(X.shape[0], 80, replace=False)
    contaminated[rows] += rng.normal(scale=25.0, size=(rows.size, X.shape[1]))

    robust = rc.TwoScatterICA(radial_clip_quantile=0.90).fit(contaminated)
    assert rc.minimum_distance_index(robust.unmixing_, mixing) < 0.16


def test_sobi_recovers_temporally_distinct_sources():
    X, _, mixing = _ar_sources(seed=8)
    estimator = rc.SOBI(lags=12).fit(X)
    assert estimator.converged_
    assert estimator.off_diagonal_energy_ < estimator.initial_off_diagonal_energy_
    assert rc.minimum_distance_index(estimator.unmixing_, mixing) < 0.08
    np.testing.assert_allclose(
        estimator.inverse_transform(estimator.transform(X)), X, rtol=1e-9, atol=1e-9
    )


def test_robust_sobi_resists_temporal_impulses_better_than_classical_sobi():
    X, _, mixing = _ar_sources(seed=9)
    rng = np.random.default_rng(99)
    contaminated = X.copy()
    rows = rng.choice(X.shape[0], 50, replace=False)
    contaminated[rows] += rng.normal(scale=30.0, size=(rows.size, X.shape[1]))

    classical = rc.SOBI(lags=12).fit(contaminated)
    robust = rc.RobustSOBI(lags=12).fit(contaminated)
    classical_error = rc.minimum_distance_index(classical.unmixing_, mixing)
    robust_error = rc.minimum_distance_index(robust.unmixing_, mixing)
    assert robust_error < 0.06
    assert robust_error < 0.25 * classical_error


def test_source_separation_estimators_clone_and_validate_rank():
    for estimator in (rc.TwoScatterICA(), rc.SOBI(), rc.RobustSOBI()):
        cloned = clone(estimator)
        assert cloned.get_params(deep=False) == estimator.get_params(deep=False)

    X = np.ones((30, 3))
    with pytest.raises(ValueError, match="rank deficient"):
        rc.SOBI(n_components=3).fit(X)
