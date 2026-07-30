"""Temporary NumPy reference check for the native FastMCD C-step."""

import numpy as np
import pytest

import robustcov as rc


pytestmark = [pytest.mark.unit, pytest.mark.native]


def _covariance_with_relative_ridge(X, support, ridge=1e-9):
    subset = X[np.asarray(support, dtype=np.int64)]
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


def _next_c_step_support(X, location, covariance, h):
    centered = X - location
    precision = np.linalg.inv(covariance)
    distances = np.maximum(
        np.einsum("ij,jk,ik->i", centered, precision, centered), 0.0
    )
    return np.argpartition(distances, h - 1)[:h]


def test_fast_mcd_selected_subset_matches_numpy_c_step_reference():
    # The final two features are nearly linear combinations of the first. This
    # exercises the relative ridge without making the example hard to inspect.
    x = np.array(
        [-2.4, -1.8, -1.2, -0.7, -0.2, 0.3, 0.8, 1.4, 2.0, 2.6, 7.5, -8.0]
    )
    X = np.column_stack(
        [
            x,
            2.0 * x
            + np.array(
                [
                    0.03,
                    -0.02,
                    0.01,
                    -0.01,
                    0.02,
                    -0.03,
                    0.01,
                    0.02,
                    -0.02,
                    0.03,
                    1.1,
                    -1.3,
                ]
            ),
            -0.5 * x
            + np.array(
                [
                    -0.02,
                    0.01,
                    0.03,
                    -0.01,
                    0.02,
                    0.00,
                    -0.03,
                    0.01,
                    0.02,
                    -0.02,
                    -0.8,
                    0.9,
                ]
            ),
        ]
    )

    fitted = rc.FastMCD(
        n_init=30,
        n_best=5,
        initial_c_steps=2,
        max_iter=50,
        tol=1e-8,
        reweight=False,
        random_state=0,
        n_jobs=1,
    ).fit(X)

    support = np.flatnonzero(fitted.raw_support_)
    expected_location, expected_covariance = _covariance_with_relative_ridge(
        X, support
    )
    native_uncorrected_covariance = fitted.raw_covariance_ / fitted.raw_scale_

    assert support.size == fitted.h_
    assert fitted.converged_ is True
    np.testing.assert_allclose(
        fitted.raw_location_, expected_location, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        native_uncorrected_covariance,
        expected_covariance,
        rtol=1e-11,
        atol=1e-12,
    )
    assert fitted.c_step_objective_value_ == pytest.approx(
        np.linalg.slogdet(expected_covariance)[1], rel=1e-10, abs=1e-10
    )

    next_support = _next_c_step_support(
        X, expected_location, expected_covariance, fitted.h_
    )
    next_location, next_covariance = _covariance_with_relative_ridge(X, next_support)
    assert np.array_equal(np.sort(next_support), support)
    assert np.linalg.slogdet(next_covariance)[1] <= (
        fitted.c_step_objective_value_ + 1e-10
    )
    np.testing.assert_allclose(
        next_location, expected_location, rtol=1e-12, atol=1e-12
    )
