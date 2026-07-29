from __future__ import annotations

import numpy as np
import pytest

import robustcov.experimental as experimental


def _relative_frobenius(estimate, truth):
    return np.linalg.norm(estimate - truth, ord="fro") / np.linalg.norm(
        truth, ord="fro"
    )


def _problem(seed=0, *, contaminated=True):
    rng = np.random.default_rng(seed)
    n, p = 700, 8
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    covariance = Q @ np.diag(np.geomspace(3.0, 0.5, p)) @ Q.T
    X = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    outliers = np.zeros(n, dtype=bool)
    if contaminated:
        indices = rng.choice(n, size=70, replace=False)
        outliers[indices] = True
        direction = rng.normal(size=p)
        direction /= np.linalg.norm(direction)
        signs = rng.choice([-1.0, 1.0], size=indices.size)
        X[indices] = (
            signs[:, None] * 10.0 * direction
            + rng.normal(scale=0.3, size=(indices.size, p))
        )
    return X, covariance, outliers


def _estimator(**kwargs):
    params = dict(
        contamination=0.1,
        max_iter=20,
        power_iterations=15,
        n_starts=2,
        random_state=0,
    )
    params.update(kwargs)
    return experimental.SpectralFilteringCovariance(**params)


def test_adversarial_filtering_improves_covariance_and_identifies_attack_rows():
    X, truth, outliers = _problem()
    empirical = np.cov(X, rowvar=False, bias=True)
    estimator = _estimator().fit(X)

    empirical_error = _relative_frobenius(empirical, truth)
    filtered_error = _relative_frobenius(estimator.covariance_, truth)
    removed_recall = np.mean(~estimator.support_[outliers])

    assert filtered_error < 0.25 * empirical_error
    assert removed_recall >= 0.85
    assert estimator.n_removed_ <= int(0.1 * X.shape[0])
    assert estimator.covariance_.shape == truth.shape
    assert np.all(np.linalg.eigvalsh(estimator.covariance_) > 0.0)
    assert estimator.stopping_reason_ in {
        "operator_within_tolerance",
        "removal_budget_exhausted",
        "no_extreme_filter_scores",
    }


def test_clean_gaussian_is_not_filtered_by_default_tolerance():
    X, truth, _ = _problem(seed=2, contaminated=False)
    estimator = _estimator().fit(X)
    empirical = np.cov(X, rowvar=False, bias=True)

    assert estimator.n_removed_ == 0
    assert estimator.converged_
    assert estimator.stopping_reason_ == "operator_within_tolerance"
    assert _relative_frobenius(estimator.covariance_, truth) <= (
        _relative_frobenius(empirical, truth) + 0.08
    )


def test_scores_labels_history_and_parameter_protocol():
    X, _, _ = _problem(seed=3)
    estimator = _estimator().fit(X)

    assert estimator.mahalanobis(X[:5]).shape == (5,)
    assert estimator.score_samples(X[:5]).shape == (5,)
    assert set(np.unique(estimator.predict(X[:20]))).issubset({-1, 1})
    assert estimator.get_params(deep=False)["contamination"] == 0.1
    assert estimator.set_params(score_threshold=7.0) is estimator
    records = estimator.history_records()
    assert records
    assert records[0]["support_size"] == X.shape[0]
    assert "operator_eigenvalue" in records[0]


def test_fit_is_deterministic_for_fixed_random_state():
    X, _, _ = _problem(seed=4)
    first = _estimator(random_state=12).fit(X)
    second = _estimator(random_state=12).fit(X)

    assert np.array_equal(first.support_, second.support_)
    assert np.allclose(first.location_, second.location_)
    assert np.allclose(first.covariance_, second.covariance_)


def test_median_imputation_and_feature_validation():
    X, _, _ = _problem(seed=5)
    X[0, 0] = np.nan
    estimator = _estimator(missing_values="median").fit(X)
    assert np.isfinite(estimator.covariance_).all()
    assert estimator.impute_values_.shape == (X.shape[1],)

    with pytest.raises(ValueError, match="features"):
        estimator.mahalanobis(np.ones((10, X.shape[1] + 1)))


def test_unfitted_and_bad_query_inputs():
    estimator = _estimator()
    with pytest.raises(AttributeError, match="not fitted"):
        estimator.mahalanobis(np.ones((3, 2)))

    X, _, _ = _problem(seed=6)
    estimator.fit(X)
    bad = X[:3].copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        estimator.mahalanobis(bad)
    with pytest.raises(ValueError, match="alpha"):
        estimator.predict(X[:3], alpha=1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"contamination": -0.1}, "contamination"),
        ({"contamination": 0.4}, "contamination"),
        ({"max_iter": 0}, "max_iter"),
        ({"filter_strength": 0.0}, "filter_strength"),
        ({"score_threshold": 0.0}, "score_threshold"),
        ({"removal_fraction": 1.0}, "removal_fraction"),
        ({"shrinkage": 0.0}, "shrinkage"),
        ({"n_starts": 0}, "n_starts"),
        ({"scale_correction": "mean"}, "scale_correction"),
        ({"missing_values": "ignore"}, "missing_values"),
    ],
)
def test_parameter_validation(kwargs, message):
    X, _, _ = _problem(seed=7, contaminated=False)
    with pytest.raises((TypeError, ValueError), match=message):
        _estimator(**kwargs).fit(X)


def test_requires_minimum_rows():
    with pytest.raises(ValueError, match="at least 8"):
        _estimator().fit(np.ones((7, 3)))


def test_sklearn_clone_when_available():
    sklearn = pytest.importorskip("sklearn")
    estimator = experimental.SpectralFilteringCovariance(
        contamination=0.08,
        filter_strength=9.0,
    )
    clone = sklearn.base.clone(estimator)
    assert clone.contamination == 0.08
    assert clone.filter_strength == 9.0
