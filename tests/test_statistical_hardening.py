from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

import robustcov as rc


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def _correlated_sample(seed: int = 0, n: int = 160, p: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(p)
    covariance = 0.6 ** np.abs(indices[:, None] - indices[None, :])
    return rng.multivariate_normal(np.linspace(-0.5, 0.5, p), covariance, size=n)


def _mcd_subset_covariance(
    X: np.ndarray, support: np.ndarray, ridge: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    subset = X[np.asarray(support)]
    location = subset.mean(axis=0)
    centered = subset - location
    covariance = centered.T @ centered / (len(subset) - 1)
    average_variance = np.trace(covariance) / X.shape[1]
    effective_ridge = (
        ridge * average_variance
        if np.isfinite(average_variance)
        and average_variance > np.finfo(np.float64).tiny
        else ridge
    )
    covariance = covariance.copy()
    covariance.flat[:: X.shape[1] + 1] += effective_ridge
    return location, covariance


def _mcd_consistency_factor(alpha: float, n_features: int) -> float:
    quantile = chi2.ppf(alpha, n_features)
    return float(alpha / chi2.cdf(quantile, n_features + 2))


def test_fast_mcd_is_affine_and_unit_equivariant_with_relative_native_ridge():
    X = _correlated_sample(seed=11, n=180, p=4)
    X[:20] += np.array([7.0, -5.0, 4.0, 6.0])
    transform = np.array(
        [[1.3, 0.2, 0.0, 0.1], [0.1, 0.9, 0.2, 0.0], [0.0, 0.3, 1.1, 0.1], [0.2, 0.0, 0.1, 1.0]]
    )
    shift = np.array([3.0, -2.0, 0.5, 1.0])

    kwargs = dict(n_init=60, n_best=6, random_state=0, reweight=True)
    base = rc.FastMCD(**kwargs).fit(X)
    fitted = rc.FastMCD(**kwargs).fit(X @ transform.T + shift)

    np.testing.assert_allclose(fitted.location_, transform @ base.location_ + shift, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(
        fitted.covariance_, transform @ base.covariance_ @ transform.T, rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(fitted.distances_, base.distances_, rtol=2e-7, atol=2e-8)

    tiny = 1e-100
    tiny_fit = rc.FastMCD(**kwargs).fit(tiny * X)
    np.testing.assert_allclose(tiny_fit.location_, tiny * base.location_, rtol=2e-8, atol=1e-110)
    np.testing.assert_allclose(tiny_fit.covariance_, tiny**2 * base.covariance_, rtol=2e-7, atol=1e-210)
    np.testing.assert_allclose(tiny_fit.distances_, base.distances_, rtol=2e-7, atol=2e-8)


def test_fast_mcd_native_solution_matches_numpy_reference():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(120, 2))
    X = np.column_stack(
        [
            latent[:, 0],
            2.0 * latent[:, 0] + 1e-3 * latent[:, 1],
            -0.5 * latent[:, 0] + 0.2 * latent[:, 1],
        ]
    )
    X[:15] += np.array([7.0, -6.0, 5.0])
    reweight_alpha = 0.975

    fitted = rc.FastMCD(
        n_init=40,
        n_best=5,
        initial_c_steps=2,
        max_iter=50,
        tol=1e-8,
        reweight=True,
        reweight_alpha=reweight_alpha,
        random_state=0,
        n_jobs=1,
    ).fit(X)

    raw_location, raw_covariance = _mcd_subset_covariance(
        X, fitted.raw_support_
    )
    raw_factor = _mcd_consistency_factor(fitted.h_ / len(X), X.shape[1])
    cutoff = float(chi2.ppf(reweight_alpha, X.shape[1]))
    final_factor = _mcd_consistency_factor(reweight_alpha, X.shape[1])

    np.testing.assert_allclose(fitted.raw_location_, raw_location, atol=1e-12)
    np.testing.assert_allclose(
        fitted.raw_covariance_, raw_factor * raw_covariance, rtol=1e-11, atol=1e-12
    )
    assert fitted.c_step_objective_value_ == pytest.approx(
        np.linalg.slogdet(raw_covariance)[1], rel=1e-8, abs=1e-6
    )
    assert fitted.raw_consistency_factor_ == pytest.approx(raw_factor)
    assert fitted.reweight_threshold_ == pytest.approx(cutoff)
    assert fitted.consistency_factor_ == pytest.approx(final_factor)
    np.testing.assert_array_equal(
        fitted.support_, fitted.raw_distances_ <= cutoff
    )

    precision = np.linalg.inv(raw_covariance)
    centered = X - raw_location
    distances = np.maximum(
        np.einsum("ij,jk,ik->i", centered, precision, centered), 0.0
    )
    next_support = np.argpartition(distances, fitted.h_ - 1)[: fitted.h_]
    np.testing.assert_array_equal(
        np.sort(next_support), np.flatnonzero(fitted.raw_support_)
    )

    final_location, final_covariance = _mcd_subset_covariance(
        X, fitted.support_
    )
    np.testing.assert_allclose(fitted.location_, final_location, atol=1e-12)
    np.testing.assert_allclose(
        fitted.covariance_, final_factor * final_covariance, rtol=1e-11, atol=1e-12
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: rc.RegularizedCauchy(alpha=0.15, max_iter=300, tol=1e-8, warn_on_nonconvergence=False),
        lambda: rc.StudentTScatter(df=3.0, alpha=0.15, max_iter=300, tol=1e-8, warn_on_nonconvergence=False),
    ],
)
def test_iterative_m_scatter_is_equivariant_to_global_measurement_units(factory):
    X = _correlated_sample(seed=12, n=150, p=5)
    X[:15] += 5.0
    factor = 2.5e-4

    base = factory().fit(X)
    scaled = factory().fit(factor * X)

    np.testing.assert_allclose(scaled.location_, factor * base.location_, rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(scaled.covariance_, factor**2 * base.covariance_, rtol=2e-6, atol=2e-12)
    np.testing.assert_allclose(scaled.distances_, base.distances_, rtol=2e-6, atol=2e-7)
    assert scaled.scale_ == pytest.approx(factor**2 * base.scale_, rel=2e-6)


def test_regularized_tyler_shape_is_stable_at_extreme_units():
    X = _correlated_sample(seed=13, n=140, p=4)
    base = rc.RegularizedTyler(alpha=0.2, max_iter=300, tol=1e-9).fit(X)
    tiny = rc.RegularizedTyler(alpha=0.2, max_iter=300, tol=1e-9).fit(1e-100 * X)
    np.testing.assert_allclose(tiny.shape_, base.shape_, rtol=2e-8, atol=2e-9)


def test_unregularized_tyler_rejects_rank_deficient_or_concentrated_data():
    rng = np.random.default_rng(14)
    rank_deficient = np.c_[rng.normal(size=(100, 3)), np.ones(100)]
    with pytest.raises(ValueError, match="RegularizedTyler"):
        rc.TylerShape().fit(rank_deficient)

    concentrated = np.vstack([rng.normal(size=(20, 4)), np.zeros((60, 4))])
    with pytest.raises(ValueError, match="RegularizedTyler"):
        rc.TylerShape(max_iter=200).fit(concentrated)

    regularized = rc.RegularizedTyler(alpha=0.2, max_iter=200).fit(rank_deficient)
    assert np.all(np.linalg.eigvalsh(regularized.covariance_) > 0.0)


@pytest.mark.parametrize("factory", [lambda: rc.DetS(max_iter=80), lambda: rc.DetMM(max_iter=80)])
def test_deterministic_s_family_handles_constant_columns_with_spd_output(factory):
    rng = np.random.default_rng(15)
    X = np.c_[rng.normal(size=(120, 3)), np.ones(120)]
    fitted = factory().fit(X)
    eigenvalues = np.linalg.eigvalsh(fitted.covariance_)
    assert np.all(np.isfinite(fitted.covariance_))
    assert eigenvalues[0] > 0.0
    assert np.all(np.isfinite(fitted.distances_))


def test_robust_standardization_preserves_tiny_measurement_units():
    X = _correlated_sample(seed=16, n=150, p=4)
    factor = 1e-100

    base_mrcd = rc.MRCD(n_init=12, n_best=4, max_iter=30, random_state=0).fit(X)
    tiny_mrcd = rc.MRCD(n_init=12, n_best=4, max_iter=30, random_state=0).fit(factor * X)
    np.testing.assert_allclose(tiny_mrcd.location_, factor * base_mrcd.location_, rtol=2e-8, atol=1e-110)
    np.testing.assert_allclose(
        tiny_mrcd.covariance_, factor**2 * base_mrcd.covariance_, rtol=2e-8, atol=1e-210
    )
    np.testing.assert_allclose(tiny_mrcd.distances_, base_mrcd.distances_, rtol=2e-8, atol=2e-8)

    base_dets = rc.DetS(max_iter=80).fit(X)
    tiny_dets = rc.DetS(max_iter=80).fit(factor * X)
    np.testing.assert_allclose(tiny_dets.location_, factor * base_dets.location_, rtol=3e-6, atol=1e-108)
    np.testing.assert_allclose(
        tiny_dets.covariance_, factor**2 * base_dets.covariance_, rtol=3e-6, atol=1e-208
    )


def test_fast_mcd_matches_sklearn_reference_on_contaminated_gaussian_data():
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    MinCovDet = sklearn_covariance.MinCovDet

    rng = np.random.default_rng(17)
    p = 5
    indices = np.arange(p)
    covariance = 0.55 ** np.abs(indices[:, None] - indices[None, :])
    X = rng.multivariate_normal(np.zeros(p), covariance, size=220)
    X[:30] += np.linspace(4.0, 8.0, p)

    ours = rc.FastMCD(contamination=0.15, n_init=100, n_best=8, random_state=0).fit(X)
    reference = MinCovDet(support_fraction=0.85, random_state=0).fit(X)

    location_scale = np.sqrt(np.trace(reference.covariance_))
    assert np.linalg.norm(ours.location_ - reference.location_) / location_scale < 0.02
    assert _relative_error(ours.covariance_, reference.covariance_) < 0.10
    assert np.corrcoef(ours.distances_, reference.mahalanobis(X))[0, 1] > 0.995


def test_fast_mcd_contamination_curve_remains_bounded_against_empirical_covariance():
    rng = np.random.default_rng(18)
    n, p = 240, 5
    indices = np.arange(p)
    covariance = 0.6 ** np.abs(indices[:, None] - indices[None, :])
    clean = rng.multivariate_normal(np.zeros(p), covariance, size=n)

    robust_errors = []
    empirical_errors = []
    for fraction in (0.10, 0.20, 0.30):
        X = clean.copy()
        count = int(round(fraction * n))
        X[:count] += rng.normal(loc=8.0, scale=1.0, size=(count, p))
        empirical_errors.append(_relative_error(np.cov(X, rowvar=False), covariance))
        fitted = rc.FastMCD(
            contamination=min(0.35, fraction + 0.05), n_init=80, n_best=8, random_state=0
        ).fit(X)
        robust_errors.append(_relative_error(fitted.covariance_, covariance))

    assert max(robust_errors) < 0.25
    assert all(robust < 0.1 * empirical for robust, empirical in zip(robust_errors, empirical_errors))
