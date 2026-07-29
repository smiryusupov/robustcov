from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

import robustcov as rc


def _factor_sample(seed=0, n_samples=500, n_features=15, n_factors=3):
    rng = np.random.default_rng(seed)
    loadings, _ = np.linalg.qr(rng.normal(size=(n_features, n_factors)))
    factors = rng.standard_t(4, size=(n_samples, n_factors))
    noise = 0.25 * rng.normal(size=(n_samples, n_features))
    clean = factors @ loadings.T + noise
    contaminated = clean.copy()
    rows = rng.choice(n_samples, 30, replace=False)
    contaminated[rows] += rng.normal(scale=8.0, size=(rows.size, n_features))
    return clean, contaminated, loadings


def _subspace_error(estimated, truth):
    estimated_projection = estimated @ estimated.T
    truth_projection = truth @ truth.T
    return np.linalg.norm(estimated_projection - truth_projection, ord="fro") / np.sqrt(
        2.0 * truth.shape[1]
    )


def test_spatial_kendall_matrix_is_scale_and_permutation_equivariant():
    rng = np.random.default_rng(2)
    X = rng.standard_t(3, size=(250, 6))
    base = rc.spatial_kendall_matrix(X, max_pairs=10000, random_state=0)
    tiny = rc.spatial_kendall_matrix(1e-200 * X, max_pairs=10000, random_state=0)
    np.testing.assert_allclose(tiny, base, rtol=1e-12, atol=1e-12)

    permutation = np.array([3, 0, 5, 2, 1, 4])
    permuted = rc.spatial_kendall_matrix(
        X[:, permutation], max_pairs=10000, random_state=0
    )
    np.testing.assert_allclose(
        permuted, base[np.ix_(permutation, permutation)], rtol=1e-12, atol=1e-12
    )


def test_kendall_factor_model_selects_factor_count_and_recovers_subspace():
    _, contaminated, loadings = _factor_sample(seed=3)
    estimator = rc.RobustFactorModel(
        n_factors="auto",
        method="kendall",
        max_factors=6,
        max_pairs=20000,
        random_state=0,
    ).fit(contaminated)

    assert estimator.n_factors_ == 3
    assert _subspace_error(estimator.loadings_, loadings) < 0.08
    assert estimator.common_component_.shape == contaminated.shape
    assert estimator.idiosyncratic_.shape == contaminated.shape
    assert np.linalg.eigvalsh(estimator.covariance_).min() > 0.0
    np.testing.assert_allclose(
        estimator.inverse_transform(estimator.transform(contaminated)),
        estimator.common_component_,
        rtol=1e-8,
        atol=1e-8,
    )


def test_huber_factor_model_recovers_contaminated_loading_space():
    _, contaminated, loadings = _factor_sample(seed=4)
    estimator = rc.RobustFactorModel(
        n_factors=3,
        method="huber",
        max_pairs=20000,
        max_iter=60,
        tol=1e-5,
        random_state=0,
    ).fit(contaminated)

    assert _subspace_error(estimator.loadings_, loadings) < 0.15
    assert estimator.n_iter_ <= 60
    assert np.isfinite(estimator.objective_)
    np.testing.assert_allclose(
        estimator.common_component_ + estimator.idiosyncratic_,
        contaminated,
        rtol=1e-10,
        atol=1e-10,
    )


def test_factor_model_is_cloneable_and_rejects_coincident_data():
    estimator = rc.RobustFactorModel(n_factors=2, method="huber")
    cloned = clone(estimator)
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)

    with pytest.raises(ValueError, match="coincident"):
        rc.RobustFactorModel(n_factors=1).fit(np.ones((20, 4)))
