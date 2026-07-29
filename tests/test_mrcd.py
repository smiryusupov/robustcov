import numpy as np
import pytest

import robustcov as rc


def _fit_fast(**kwargs):
    defaults = dict(
        quality="fast",
        n_init=10,
        n_best=4,
        initial_c_steps=2,
        max_iter=40,
        random_state=0,
    )
    defaults.update(kwargs)
    return rc.MRCD(**defaults)


def test_mrcd_fits_high_dimensional_data_and_is_spd():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(42, 70))
    X[:5, :8] += 9.0

    est = _fit_fast(contamination=0.15).fit(X)

    assert est.covariance_.shape == (70, 70)
    assert est.precision_.shape == (70, 70)
    assert est.support_.sum() == est.h_ == 36
    assert est.regularization_ > 0.0
    assert np.linalg.eigvalsh(est.covariance_).min() > 0.0
    assert est.standardized_condition_number_ <= est.max_condition_number * 1.001
    assert np.isfinite(est.distances_).all()


def test_mrcd_resists_shifted_row_outliers():
    rng = np.random.default_rng(2)
    clean = rng.normal(size=(120, 8))
    outliers = rng.normal(loc=12.0, scale=0.5, size=(20, 8))
    X = np.vstack([clean, outliers])

    est = _fit_fast(contamination=0.16, n_init=20).fit(X)

    empirical_error = np.linalg.norm(X.mean(axis=0))
    robust_error = np.linalg.norm(est.location_)
    assert robust_error < 0.25 * empirical_error
    assert np.count_nonzero(est.support_[120:]) <= 2
    assert np.median(est.distances_[120:]) > np.median(est.distances_[:120])


def test_mrcd_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(3)
    X = rng.standard_t(df=4, size=(75, 18))

    first = _fit_fast(random_state=9).fit(X)
    second = _fit_fast(random_state=9).fit(X)

    np.testing.assert_array_equal(first.support_, second.support_)
    np.testing.assert_allclose(first.location_, second.location_)
    np.testing.assert_allclose(first.covariance_, second.covariance_)
    assert first.regularization_ == pytest.approx(second.regularization_)


def test_mrcd_scale_and_location_equivariance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 10))
    scales = np.linspace(0.5, 3.0, X.shape[1])
    shift = np.linspace(-2.0, 2.0, X.shape[1])
    Y = X * scales + shift

    est_x = _fit_fast(n_init=16).fit(X)
    est_y = _fit_fast(n_init=16).fit(Y)

    np.testing.assert_array_equal(est_x.support_, est_y.support_)
    np.testing.assert_allclose(est_y.location_, est_x.location_ * scales + shift, rtol=1e-8, atol=1e-8)
    expected_covariance = np.diag(scales) @ est_x.covariance_ @ np.diag(scales)
    np.testing.assert_allclose(est_y.covariance_, expected_covariance, rtol=1e-7, atol=1e-8)


def test_mrcd_custom_and_equicorrelation_targets():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(70, 12))
    custom = 0.25 * np.ones((12, 12)) + 0.75 * np.eye(12)

    custom_est = _fit_fast(target=custom, regularization=0.2).fit(X)
    equi_est = _fit_fast(target="equicorrelation").fit(X)

    assert custom_est.target_name_ == "custom"
    assert custom_est.regularization_ == pytest.approx(0.2)
    assert np.linalg.eigvalsh(custom_est.target_).min() > 0
    assert equi_est.target_name_ == "equicorrelation"
    assert equi_est.target_correlation_ is not None
    assert -1.0 / 11 < equi_est.target_correlation_ < 1.0


def test_mrcd_objective_does_not_increase_during_c_steps():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(90, 14))
    X[:8] += 6.0

    est = _fit_fast(n_init=20, max_iter=60).fit(X)
    differences = np.diff(est.objective_path_)
    assert np.all(differences <= 1e-8)


def test_mrcd_support_fraction_one_and_one_dimensional_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 1))

    est = _fit_fast(support_fraction=1.0, n_init=3).fit(X)

    assert est.h_ == X.shape[0]
    assert est.support_.all()
    assert est.covariance_.shape == (1, 1)
    assert est.covariance_[0, 0] > 0


def test_mrcd_missing_value_median_imputation():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 9))
    X[0, 0] = np.nan
    X[4, 5] = np.nan

    est = _fit_fast(missing_values="median").fit(X)

    assert np.isfinite(est.covariance_).all()
    assert np.isfinite(est.impute_values_).all()


def test_mrcd_integrates_with_pca_and_feature_geometry():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(65, 25))
    estimator = _fit_fast(n_init=12)

    pca = rc.RobustPCA(n_components=5, estimator=estimator).fit(X)
    geometry = rc.FeatureGeometry(estimator=estimator).fit(X)

    assert pca.transform(X).shape == (65, 5)
    assert np.isfinite(pca.orthogonal_distances(X)).all()
    assert geometry.transform(X).shape == X.shape
    assert np.isfinite(geometry.mahalanobis_scores(X)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_fraction": 0.49},
        {"contamination": 0.5},
        {"regularization": -0.1},
        {"regularization": 1.1},
        {"max_condition_number": 1.0},
        {"standardization": "std"},
        {"target": "unknown"},
    ],
)
def test_mrcd_rejects_invalid_parameters(kwargs):
    X = np.arange(30, dtype=float).reshape(10, 3)
    with pytest.raises(ValueError):
        _fit_fast(**kwargs).fit(X)



def test_zero_regularization_returns_the_selected_subset_covariance():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 5))
    X[:8] += 5.0

    est = _fit_fast(
        contamination=0.10,
        regularization=0.0,
        n_init=20,
    ).fit(X)

    selected = X[est.support_]
    expected_location = selected.mean(axis=0)
    expected_covariance = est.consistency_factor_ * np.cov(
        selected, rowvar=False, ddof=1
    )
    np.testing.assert_allclose(est.location_, expected_location, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        est.covariance_, expected_covariance, rtol=1e-8, atol=1e-9
    )


def test_auto_regularization_can_switch_off_for_well_conditioned_low_dimensional_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(140, 4))

    est = _fit_fast(support_fraction=0.80, n_init=12).fit(X)

    assert est.regularization_ == pytest.approx(0.0)
    assert est.standardized_condition_number_ < est.max_condition_number

def test_zero_regularization_rejected_when_p_is_at_least_h():
    X = np.random.default_rng(10).normal(size=(30, 40))
    with pytest.raises(ValueError, match="p >= h"):
        _fit_fast(regularization=0.0).fit(X)
