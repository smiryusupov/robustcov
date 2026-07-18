import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import robustcov as rc


def _fit_fast(**kwargs):
    defaults = dict(
        n_init=12,
        n_best=4,
        initial_c_steps=2,
        max_iter=35,
        random_state=0,
    )
    defaults.update(kwargs)
    return rc.KMRCD(**defaults)


def test_linear_kernel_matches_explicit_feature_space_distance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 5))
    X[:8] += 5.0

    est = _fit_fast(
        kernel="linear",
        contamination=0.10,
        regularization=0.2,
    ).fit(X)

    U = est.X_fit_standardized_
    subset = U[est.support_]
    center = subset.mean(axis=0)
    covariance = np.cov(subset, rowvar=False, ddof=1)
    regularized = (1.0 - est.regularization_) * covariance + est.regularization_ * np.eye(U.shape[1])
    centered = U - center
    expected = np.einsum("ij,jk,ik->i", centered, np.linalg.inv(regularized), centered)

    np.testing.assert_allclose(est.distances_, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(est.mahalanobis(X), expected, rtol=1e-8, atol=1e-8)


def test_precomputed_linear_kernel_matches_coordinate_fit():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(75, 6))
    coordinate = _fit_fast(kernel="linear", contamination=0.12, regularization=0.15).fit(X)
    U = coordinate.X_fit_standardized_
    K = U @ U.T

    precomputed = _fit_fast(kernel="precomputed", contamination=0.12, regularization=0.15).fit(K)

    np.testing.assert_array_equal(coordinate.support_, precomputed.support_)
    np.testing.assert_allclose(coordinate.distances_, precomputed.distances_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        precomputed.mahalanobis(K, kernel_diag=np.diag(K)),
        precomputed.distances_,
        rtol=1e-9,
        atol=1e-9,
    )


def test_rbf_kernel_detects_off_manifold_outliers_better_than_linear_mrcd():
    rng = np.random.default_rng(3)
    n_inliers = 220
    x = rng.uniform(-2.5, 2.5, n_inliers)
    inliers = np.column_stack([x, 0.55 * x**2 + 0.08 * rng.normal(size=n_inliers)])

    n_outliers = 45
    xo = rng.uniform(-1.8, 1.8, n_outliers)
    yo = rng.uniform(0.2, 2.0, n_outliers)
    close = np.abs(yo - 0.55 * xo**2) < 0.4
    while np.any(close):
        yo[close] = rng.uniform(0.2, 2.0, np.count_nonzero(close))
        close = np.abs(yo - 0.55 * xo**2) < 0.4
    outliers = np.column_stack([xo, yo])
    X = np.vstack([inliers, outliers])
    labels = np.r_[np.zeros(n_inliers), np.ones(n_outliers)]

    kernel = _fit_fast(kernel="rbf", gamma=2.0, contamination=0.18, n_init=20).fit(X)
    linear = rc.MRCD(
        contamination=0.18,
        n_init=20,
        n_best=4,
        initial_c_steps=2,
        max_iter=35,
        random_state=0,
    ).fit(X)

    kernel_auc = roc_auc_score(labels, kernel.distances_)
    linear_auc = roc_auc_score(labels, linear.distances_)
    assert kernel_auc > 0.90
    assert kernel_auc > linear_auc + 0.25


def test_training_and_out_of_sample_rbf_scores_agree():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(70, 4))
    est = _fit_fast(kernel="rbf", gamma="median", contamination=0.10).fit(X)
    np.testing.assert_allclose(est.mahalanobis(X), est.distances_, rtol=1e-8, atol=1e-8)


def test_fixed_seed_is_deterministic():
    rng = np.random.default_rng(5)
    X = rng.standard_t(df=4, size=(80, 7))
    first = _fit_fast(kernel="rbf", gamma="median", random_state=12).fit(X)
    second = _fit_fast(kernel="rbf", gamma="median", random_state=12).fit(X)
    np.testing.assert_array_equal(first.support_, second.support_)
    np.testing.assert_allclose(first.distances_, second.distances_)
    assert first.regularization_ == pytest.approx(second.regularization_)


def test_objective_is_nonincreasing_and_model_is_well_conditioned():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(85, 8))
    X[:10] += 4.0
    est = _fit_fast(kernel="rbf", gamma="scale", contamination=0.15, n_init=20).fit(X)
    assert np.all(np.diff(est.objective_path_) <= 1e-8)
    assert est.standardized_condition_number_ <= est.max_condition_number * 1.001
    assert np.linalg.eigvalsh(est.regularized_kernel_).min() > 0


def test_predict_and_decision_function_use_fitted_cutoff():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(70, 3))
    X = np.vstack([X, np.full((5, 3), 8.0)])
    est = _fit_fast(kernel="rbf", gamma=0.5, contamination=0.10).fit(X)
    prediction = est.predict(X)
    decision = est.decision_function(X)
    np.testing.assert_array_equal(prediction, np.where(decision >= 0, 1, -1))
    assert np.isfinite(est.distance_threshold_)
    assert est.distance_threshold_ > 0


def test_missing_value_median_imputation():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 5))
    X[0, 0] = np.nan
    X[3, 2] = np.nan
    est = _fit_fast(kernel="rbf", missing_values="median").fit(X)
    assert np.isfinite(est.kernel_matrix_).all()
    assert np.isfinite(est.mahalanobis(X)).all()


def test_polynomial_kernel_and_callable_kernel():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(55, 3))
    poly = _fit_fast(kernel="polynomial", gamma=0.3, degree=2, coef0=1.0).fit(X)
    assert np.isfinite(poly.distances_).all()

    def linear_kernel(A, B):
        return A @ B.T

    custom = _fit_fast(kernel=linear_kernel, gamma=1.0, regularization=0.2).fit(X)
    assert np.isfinite(custom.mahalanobis(X[:4])).all()


def test_precomputed_scoring_requires_self_kernel_diagonal():
    X = np.random.default_rng(10).normal(size=(30, 3))
    K = X @ X.T
    est = _fit_fast(kernel="precomputed", regularization=0.2).fit(K)
    with pytest.raises(ValueError, match="kernel_diag"):
        est.mahalanobis(K)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_fraction": 0.49},
        {"contamination": 0.5},
        {"regularization": 0.0},
        {"regularization": 1.0},
        {"regularization": "bad"},
        {"max_condition_number": 1.0},
        {"gamma": 0.0},
        {"degree": 0},
        {"cutoff_quantile": 0.5},
        {"n_init": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    X = np.random.default_rng(11).normal(size=(25, 3))
    with pytest.raises(ValueError):
        _fit_fast(**kwargs).fit(X)


def test_invalid_precomputed_kernel_is_rejected():
    K = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        _fit_fast(kernel="precomputed", regularization=0.2).fit(K)
