import numpy as np
import pytest

import robustcov as rc
from robustcov.s_estimators import _location_efficiency, _mm_tuning_constant, _s_tuning_constant


def make_data(seed=0, n=180, p=5, contamination=0.15):
    rng = np.random.default_rng(seed)
    indices = np.arange(p)
    covariance = 0.55 ** np.abs(indices[:, None] - indices[None, :])
    X = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    count = int(round(contamination * n))
    labels = np.zeros(n, dtype=bool)
    labels[:count] = True
    direction = np.linspace(0.7, 1.4, p)
    X[:count] += 6.0 * direction
    return X, covariance, labels


def relative_error(a, b):
    return np.linalg.norm(a - b, ord="fro") / np.linalg.norm(b, ord="fro")


def test_public_aliases():
    assert rc.DetS is rc.DeterministicSEstimator
    assert rc.DetMM is rc.DeterministicMMEstimator


def test_dets_basic_fit_and_attributes():
    X, _, _ = make_data()
    model = rc.DetS(max_iter=80).fit(X)
    assert model.location_.shape == (X.shape[1],)
    assert model.covariance_.shape == (X.shape[1], X.shape[1])
    assert model.precision_.shape == model.covariance_.shape
    assert model.distances_.shape == (X.shape[0],)
    assert model.weights_.shape == (X.shape[0],)
    assert model.support_.dtype == bool
    assert np.all(np.linalg.eigvalsh(model.covariance_) > 0)
    assert model.converged_
    assert model.n_initial_models_ == 6


def test_detmm_basic_fit_and_initial_estimator():
    X, _, _ = make_data()
    model = rc.DetMM(max_iter=80).fit(X)
    assert isinstance(model.initial_estimator_, rc.DetS)
    assert model.initial_covariance_.shape == model.covariance_.shape
    assert model.s_scale_ > 0
    assert model.tuning_constant_ > model.s_tuning_constant_
    assert model.converged_
    assert np.all(np.linalg.eigvalsh(model.covariance_) > 0)


def test_deterministic_under_repeated_fit_and_row_permutation():
    X, _, _ = make_data(seed=2)
    first = rc.DetS(max_iter=80).fit(X)
    second = rc.DetS(max_iter=80).fit(X)
    assert np.allclose(first.location_, second.location_)
    assert np.allclose(first.covariance_, second.covariance_)

    permutation = np.random.default_rng(10).permutation(X.shape[0])
    permuted = rc.DetS(max_iter=80).fit(X[permutation])
    assert np.allclose(first.location_, permuted.location_, atol=2e-5, rtol=2e-5)
    assert np.allclose(first.covariance_, permuted.covariance_, atol=3e-5, rtol=3e-5)


def test_near_affine_equivariance():
    X, _, _ = make_data(seed=4, n=200, p=4)
    A = np.array(
        [[1.2, 0.3, 0.0, 0.0], [0.1, 0.9, 0.2, 0.0], [0.0, 0.2, 1.1, 0.1], [0.1, 0.0, 0.2, 1.0]]
    )
    shift = np.array([1.0, -2.0, 0.5, 0.3])
    transformed = X @ A.T + shift
    base = rc.DetS(max_iter=80).fit(X)
    fitted = rc.DetS(max_iter=80).fit(transformed)
    expected_location = A @ base.location_ + shift
    expected_covariance = A @ base.covariance_ @ A.T
    assert np.allclose(fitted.location_, expected_location, atol=2e-4, rtol=2e-4)
    assert np.allclose(fitted.covariance_, expected_covariance, atol=3e-4, rtol=3e-4)


def test_outliers_receive_small_weights_and_large_distances():
    X, _, labels = make_data(seed=5)
    model = rc.DetMM(max_iter=80).fit(X)
    assert np.median(model.weights_[labels]) < 0.05
    assert np.median(model.weights_[~labels]) > 0.5
    assert np.median(model.distances_[labels]) > 10 * np.median(model.distances_[~labels])


def test_detmm_improves_clean_gaussian_efficiency_proxy():
    rng = np.random.default_rng(1)
    p = 4
    covariance = 0.5 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    X = rng.multivariate_normal(np.zeros(p), covariance, size=300)
    dets = rc.DetS(max_iter=80).fit(X)
    detmm = rc.DetMM(max_iter=80, efficiency=0.95).fit(X)
    assert relative_error(detmm.covariance_, covariance) < relative_error(dets.covariance_, covariance)


def test_tuning_constants_match_target_definitions():
    c_s, b = _s_tuning_constant(5, 0.5)
    assert b / (c_s * c_s / 6.0) == pytest.approx(0.5, rel=1e-8)
    c_mm = _mm_tuning_constant(3, 0.95)
    assert c_mm == pytest.approx(5.49025, rel=2e-4)
    assert _location_efficiency(c_mm, 3) == pytest.approx(0.95, rel=1e-6)


def test_missing_value_median_mode():
    X, _, _ = make_data(seed=6, n=160, p=4)
    X[2, 1] = np.nan
    X[10, 3] = np.nan
    model = rc.DetS(missing_values="median", max_iter=80).fit(X)
    assert np.isfinite(model.covariance_).all()
    assert model.impute_values_.shape == (X.shape[1],)
    scores = model.mahalanobis(X[:8])
    assert np.isfinite(scores).all()


def test_works_as_robust_pca_scatter_estimator():
    X, _, _ = make_data(seed=7, n=180, p=5)
    pca = rc.RobustPCA(n_components=2, estimator=rc.DetMM(max_iter=60)).fit(X)
    assert pca.components_.shape == (2, X.shape[1])
    assert np.isfinite(pca.transform(X[:5])).all()


def test_predict_and_score_samples():
    X, _, _ = make_data(seed=8, n=160, p=4)
    model = rc.DetS(max_iter=80).fit(X)
    prediction = model.predict(X[:10])
    score = model.score_samples(X[:10])
    assert set(np.unique(prediction)).issubset({-1, 1})
    assert np.allclose(score, -0.5 * model.mahalanobis(X[:10]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"breakdown": 0.01}, "breakdown"),
        ({"breakdown": 0.6}, "breakdown"),
        ({"n_best": 0}, "n_best"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0}, "tol"),
        ({"ridge": -1}, "ridge"),
        ({"standardization": "bad"}, "standardization"),
        ({"missing_values": "bad"}, "missing_values"),
    ],
)
def test_dets_parameter_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rc.DetS(**kwargs)


def test_detmm_parameter_validation():
    with pytest.raises(ValueError, match="efficiency"):
        rc.DetMM(efficiency=0.2)
    with pytest.raises(ValueError, match="efficiency"):
        rc.DetMM(efficiency=1.0)


def test_dimension_guard_recommends_regularized_estimator():
    X = np.random.default_rng(0).normal(size=(20, 10))
    with pytest.raises(ValueError, match="MRCD"):
        rc.DetS().fit(X)
