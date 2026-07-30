"""Temporary exact-calibration check for FastMCD."""

import numpy as np
import pytest
from scipy.stats import chi2

import robustcov as rc


pytestmark = [pytest.mark.unit, pytest.mark.native]


def _covariance_with_relative_ridge(X, support, ridge=1e-9):
    subset = X[np.asarray(support, dtype=bool)]
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


def _consistency_factor(alpha, n_features):
    quantile = chi2.ppf(alpha, n_features)
    return alpha / chi2.cdf(quantile, n_features + 2)


def test_fast_mcd_uses_exact_chi_square_calibration():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 3))
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

    raw_alpha = fitted.h_ / X.shape[0]
    expected_raw_factor = _consistency_factor(raw_alpha, X.shape[1])
    expected_cutoff = chi2.ppf(reweight_alpha, X.shape[1])
    expected_reweight_factor = _consistency_factor(
        reweight_alpha, X.shape[1]
    )

    assert fitted.reweighted_ is True
    assert fitted.raw_scale_ == pytest.approx(expected_raw_factor)
    assert fitted.raw_consistency_factor_ == pytest.approx(expected_raw_factor)
    assert fitted.reweight_threshold_ == pytest.approx(expected_cutoff)
    assert fitted.consistency_factor_ == pytest.approx(expected_reweight_factor)
    np.testing.assert_array_equal(
        fitted.support_, fitted.raw_distances_ <= expected_cutoff
    )

    raw_location, raw_covariance = _covariance_with_relative_ridge(
        X, fitted.raw_support_
    )
    final_location, final_covariance = _covariance_with_relative_ridge(
        X, fitted.support_
    )
    np.testing.assert_allclose(fitted.raw_location_, raw_location, atol=1e-12)
    np.testing.assert_allclose(
        fitted.raw_covariance_,
        expected_raw_factor * raw_covariance,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(fitted.location_, final_location, atol=1e-12)
    np.testing.assert_allclose(
        fitted.covariance_,
        expected_reweight_factor * final_covariance,
        rtol=1e-11,
        atol=1e-12,
    )
