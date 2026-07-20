# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import robustcov as rc
import robustcov.m_estimators as m_estimators


def _legacy_safe_pinv(matrix):
    return np.linalg.pinv(0.5 * (matrix + matrix.T), hermitian=True)


def _legacy_mahalanobis(centered, precision):
    return np.einsum(
        "ij,jk,ik->i", centered, precision, centered, optimize=True
    )


def test_fast_precision_matches_inverse_and_singular_fallback():
    rng = np.random.default_rng(31)
    design = rng.normal(size=(9, 9))
    positive_definite = design.T @ design + 0.2 * np.eye(9)
    precision = m_estimators._safe_pinv(positive_definite)
    np.testing.assert_allclose(
        precision,
        np.linalg.inv(positive_definite),
        rtol=1e-12,
        atol=1e-12,
    )

    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    fallback = m_estimators._safe_pinv(singular)
    np.testing.assert_allclose(singular @ fallback @ singular, singular)
    assert np.isfinite(fallback).all()


def test_mahalanobis_contraction_matches_einsum_reference():
    rng = np.random.default_rng(32)
    centered = rng.normal(size=(170, 13))
    design = rng.normal(size=(13, 13))
    covariance = design.T @ design + 0.5 * np.eye(13)
    precision = np.linalg.inv(covariance)

    actual = m_estimators._mahalanobis_from_precision(centered, precision)
    expected = _legacy_mahalanobis(centered, precision)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_regularized_cauchy_matches_previous_numerical_path(monkeypatch):
    rng = np.random.default_rng(33)
    X = rng.standard_t(df=3, size=(260, 9))
    X[:18, :3] += 4.0
    kwargs = dict(
        alpha=0.1,
        max_iter=120,
        tol=1e-7,
        damping=0.5,
        warn_on_nonconvergence=False,
    )

    optimized = rc.RegularizedCauchy(**kwargs).fit(X)
    monkeypatch.setattr(m_estimators, "_safe_pinv", _legacy_safe_pinv)
    monkeypatch.setattr(
        m_estimators, "_mahalanobis_from_precision", _legacy_mahalanobis
    )
    legacy = rc.RegularizedCauchy(**kwargs).fit(X)

    assert optimized.n_iter_ == legacy.n_iter_
    assert optimized.converged_ == legacy.converged_
    np.testing.assert_allclose(
        optimized.location_, legacy.location_, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        optimized.covariance_, legacy.covariance_, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        optimized.distances_, legacy.distances_, rtol=1e-11, atol=1e-10
    )
    np.testing.assert_allclose(
        optimized.weights_, legacy.weights_, rtol=1e-11, atol=1e-12
    )
