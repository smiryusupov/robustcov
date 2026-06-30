# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import robustcov as rc


def test_feature_geometry_shapes_and_scores():
    rng = np.random.default_rng(123)
    X = rng.standard_t(df=3, size=(120, 6))
    X[:8] += 5.0

    geom = rc.FeatureGeometry(
        estimator=rc.RegularizedCauchy(alpha=0.10),
    ).fit(X)

    scores = geom.mahalanobis_scores(X)
    squared = geom.squared_mahalanobis(X)
    Z = geom.transform(X)

    assert scores.shape == (120,)
    assert squared.shape == (120,)
    assert Z.shape == X.shape
    assert geom.covariance_.shape == (6, 6)
    assert geom.precision_.shape == (6, 6)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
    assert np.allclose(scores**2, squared, atol=1e-8)


def test_feature_geometry_rbf_kernel_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(80, 4))

    geom = rc.FeatureGeometry(
        estimator=rc.RegularizedCauchy(alpha=0.10),
    ).fit(X)

    K = geom.rbf_kernel(X, length_scale=1.5)

    assert K.shape == (80, 80)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_feature_geometry_pairwise_cross_distances():
    rng = np.random.default_rng(789)
    X = rng.normal(size=(90, 5))
    Y = rng.normal(size=(20, 5))

    geom = rc.FeatureGeometry().fit(X)

    D2 = geom.pairwise_squared_distances(Y, X)
    K = geom.rbf_kernel(Y, X)

    assert D2.shape == (20, 90)
    assert K.shape == (20, 90)
    assert np.all(np.isfinite(D2))
    assert np.all(D2 >= 0)
    assert np.all((K >= 0) & (K <= 1))


def test_class_conditional_feature_geometry_predicts_nearest_class():
    rng = np.random.default_rng(321)

    X0 = rng.normal(loc=-3.0, scale=0.6, size=(80, 5))
    X1 = rng.normal(loc=3.0, scale=0.6, size=(80, 5))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)]

    geom = rc.ClassConditionalFeatureGeometry(
        estimator=rc.FastMCD(n_init=20, random_state=0),
    ).fit(X, y)

    pred = geom.predict_nearest_class(X)
    scores = geom.ood_scores(X)
    D = geom.class_mahalanobis_scores(X)

    assert pred.shape == y.shape
    assert scores.shape == y.shape
    assert D.shape == (160, 2)
    assert np.mean(pred == y) > 0.95
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_class_conditional_feature_geometry_ood_scores_increase_for_far_points():
    rng = np.random.default_rng(654)

    X0 = rng.normal(loc=-2.0, scale=0.7, size=(100, 4))
    X1 = rng.normal(loc=2.0, scale=0.7, size=(100, 4))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]

    geom = rc.ClassConditionalFeatureGeometry(
        estimator=rc.FastMCD(n_init=20, random_state=0),
    ).fit(X, y)

    X_in = rng.normal(loc=2.0, scale=0.7, size=(60, 4))
    X_ood = rng.normal(loc=8.0, scale=0.7, size=(60, 4))

    in_scores = geom.ood_scores(X_in)
    ood_scores = geom.ood_scores(X_ood)

    assert ood_scores.mean() > in_scores.mean() + 3.0
