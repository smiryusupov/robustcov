# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import robustcov as rc


class FixedScatter:
    def __init__(self, covariance, location=None):
        self.covariance = np.asarray(covariance, dtype=float)
        self.location = location

    def fit(self, X):
        self.covariance_ = self.covariance.copy()
        self.location_ = (
            np.mean(X, axis=0)
            if self.location is None
            else np.asarray(self.location, dtype=float).copy()
        )
        return self


class EmpiricalScatter:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = np.mean(X, axis=0)
        self.covariance_ = np.cov(X, rowvar=False, ddof=1)
        return self


class RelativeEmpiricalScatter:
    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = np.mean(X, axis=0)
        centered = X - self.location_
        covariance = centered.T @ centered / X.shape[0]
        scale = max(
            float(np.trace(covariance)) / X.shape[1],
            np.finfo(float).tiny,
        )
        self.covariance_ = covariance + self.ridge * scale * np.eye(X.shape[1])
        return self


def _spd_sample(seed=0, n=160, p=5):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(p, p))
    covariance = matrix @ matrix.T + np.eye(p)
    location = rng.normal(size=p)
    X = rng.multivariate_normal(location, covariance, size=n)
    return X, location, covariance


def _monitor_data(seed=0, n=420, p=5):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    basis = np.zeros((p, 2))
    basis[0, 0] = 2.5
    basis[1, 1] = 1.5
    basis[2, :] = [0.6, -0.3]
    X = latent @ basis.T + rng.normal(scale=0.08, size=(n, p))
    return X


def _monitor(**kwargs):
    parameters = dict(
        n_components=2,
        estimator=RelativeEmpiricalScatter(),
        window_size=80,
        calibration_windows=8,
        threshold_quantile=0.99,
        sample_quantile=0.98,
        random_state=0,
    )
    parameters.update(kwargs)
    return rc.RobustSubspaceMonitor(**parameters)


def test_standardized_sparse_precision_is_feature_unit_equivariant():
    X, location, covariance = _spd_sample(seed=801)
    scales = np.array([1e-5, 2.0, 30.0, 1e3, 0.2])

    base = rc.RobustGraphicalLasso(
        alpha=0.08,
        scatter_estimator=FixedScatter(covariance, location),
        standardize=True,
        max_iter=1000,
    ).fit(X)
    transformed = rc.RobustGraphicalLasso(
        alpha=0.08,
        scatter_estimator=FixedScatter(
            covariance * np.outer(scales, scales), location * scales
        ),
        standardize=True,
        max_iter=1000,
    ).fit(X * scales)

    np.testing.assert_allclose(
        transformed.precision_ * np.outer(scales, scales),
        base.precision_,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        transformed.partial_correlation_, base.partial_correlation_, atol=1e-11
    )
    np.testing.assert_array_equal(transformed.adjacency_, base.adjacency_)
    np.testing.assert_allclose(
        transformed.mahalanobis(X[:20] * scales),
        base.mahalanobis(X[:20]),
        rtol=1e-11,
        atol=1e-11,
    )


def test_standardized_sparse_precision_handles_extreme_global_units():
    X, location, covariance = _spd_sample(seed=802)
    base = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(covariance, location),
        standardize=True,
    ).fit(X)

    for factor in (1e-50, 1e50):
        scaled = rc.RobustGraphicalLasso(
            alpha=0.0,
            scatter_estimator=FixedScatter(
                covariance * factor**2, location * factor
            ),
            standardize=True,
        ).fit(X * factor)
        np.testing.assert_allclose(
            scaled.covariance_ / factor**2, base.covariance_, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            scaled.mahalanobis(X[:12] * factor),
            base.mahalanobis(X[:12]),
            rtol=1e-11,
            atol=1e-11,
        )


def test_sparse_precision_handles_repeated_and_partial_constant_features():
    rng = np.random.default_rng(803)
    core = rng.normal(size=(140, 3))
    X = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])

    model = rc.RobustGraphicalLasso(
        alpha=0.05,
        scatter_estimator="empirical",
        standardize=True,
        max_iter=1000,
    ).fit(X)

    assert np.array_equal(model.constant_features_, [False, False, False, False, True])
    assert np.isfinite(model.precision_).all()
    assert np.linalg.eigvalsh(model.precision_).min() > 0.0
    np.testing.assert_allclose(
        model.covariance_ @ model.precision_, np.eye(X.shape[1]), rtol=1e-7, atol=1e-7
    )
    assert np.isfinite(model.score_samples(X[:10])).all()


def test_sparse_precision_rejects_undefined_or_invalid_scatter():
    constant = np.ones((30, 4))
    with pytest.raises(ValueError, match="no positive feature variance"):
        rc.RobustGraphicalLasso(
            alpha=0.0, scatter_estimator="empirical", standardize=True
        ).fit(constant)

    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        rc.RobustGraphicalLasso(
            alpha=0.0,
            scatter_estimator=FixedScatter(indefinite),
        ).fit(np.zeros((10, 2)))


def test_spatial_sign_precision_is_scale_and_permutation_invariant():
    rng = np.random.default_rng(804)
    half = rng.normal(size=(90, 6))
    shift = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
    X = np.vstack([half, -half]) + shift
    permutation = np.array([4, 1, 5, 0, 3, 2])

    base = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(X)
    tiny = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(1e-200 * X)
    permuted = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(X[:, permutation])

    np.testing.assert_allclose(tiny.location_ / 1e-200, base.location_, atol=1e-10)
    np.testing.assert_allclose(tiny.precision_, base.precision_, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(tiny.adjacency_, base.adjacency_)
    np.testing.assert_allclose(
        permuted.precision_, base.precision_[np.ix_(permutation, permutation)],
        rtol=1e-9, atol=1e-9,
    )
    np.testing.assert_array_equal(
        permuted.adjacency_, base.adjacency_[np.ix_(permutation, permutation)]
    )


def test_spatial_sign_precision_rejects_all_coincident_observations():
    with pytest.raises(ValueError, match="all observations coincide"):
        rc.SGLASSO(alpha=0.0).fit(np.ones((30, 5)))


def test_feature_geometry_is_unit_and_permutation_equivariant():
    X, location, covariance = _spd_sample(seed=805)
    factor = 1e-50
    permutation = np.array([3, 0, 4, 1, 2])

    base = rc.FeatureGeometry(
        estimator=FixedScatter(covariance, location), ridge=1e-10
    ).fit(X)
    tiny = rc.FeatureGeometry(
        estimator=FixedScatter(covariance * factor**2, location * factor),
        ridge=1e-10,
    ).fit(X * factor)
    permuted = rc.FeatureGeometry(
        estimator=FixedScatter(
            covariance[np.ix_(permutation, permutation)], location[permutation]
        ),
        ridge=1e-10,
    ).fit(X[:, permutation])

    np.testing.assert_allclose(
        tiny.squared_mahalanobis(X[:20] * factor),
        base.squared_mahalanobis(X[:20]),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.squared_mahalanobis(X[:20, permutation]),
        base.squared_mahalanobis(X[:20]),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        permuted.precision_, base.precision_[np.ix_(permutation, permutation)],
        rtol=1e-10,
        atol=1e-10,
    )


def test_feature_geometry_regularizes_singular_covariance_and_rejects_invalid_cases():
    rng = np.random.default_rng(806)
    core = rng.normal(size=(120, 2))
    X = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])
    model = rc.FeatureGeometry(estimator=EmpiricalScatter()).fit(X)

    assert np.linalg.matrix_rank(model.raw_covariance_) < X.shape[1]
    assert np.linalg.eigvalsh(model.covariance_).min() > 0.0
    np.testing.assert_allclose(
        model.covariance_ @ model.precision_, np.eye(X.shape[1]), rtol=1e-5, atol=1e-5
    )
    assert np.isfinite(model.transform(X)).all()

    with pytest.raises(ValueError, match="no positive feature variation"):
        rc.FeatureGeometry(estimator=EmpiricalScatter()).fit(np.ones((20, 3)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        rc.FeatureGeometry(
            estimator=FixedScatter([[1.0, 2.0], [2.0, 1.0]])
        ).fit(np.zeros((10, 2)))


def test_monitor_metrics_are_invariant_to_global_units_and_feature_permutation():
    X = _monitor_data(seed=807)
    batch = X[:80].copy()
    batch[:, 0] += 0.5
    permutation = np.array([3, 1, 4, 0, 2])

    base_monitor = _monitor().fit(X)
    base = base_monitor.evaluate(batch)
    tiny_monitor = _monitor().fit(1e-50 * X)
    tiny = tiny_monitor.evaluate(1e-50 * batch)
    permuted_monitor = _monitor().fit(X[:, permutation])
    permuted = permuted_monitor.evaluate(batch[:, permutation])

    names = (
        "location_shift",
        "scale_shift",
        "shape_shift",
        "max_subspace_angle",
        "score_distance_shift",
        "orthogonal_distance_shift",
        "combined_outlier_fraction",
    )
    for name in names:
        assert getattr(tiny, name) == pytest.approx(getattr(base, name), rel=1e-9, abs=1e-9)
        assert getattr(permuted, name) == pytest.approx(
            getattr(base, name), rel=1e-9, abs=1e-9
        )
    for name in base_monitor.thresholds_:
        assert tiny_monitor.thresholds_[name] == pytest.approx(
            base_monitor.thresholds_[name], rel=1e-9, abs=1e-9
        )
        assert permuted_monitor.thresholds_[name] == pytest.approx(
            base_monitor.thresholds_[name], rel=1e-9, abs=1e-9
        )


def test_monitor_handles_repeated_and_constant_reference_features():
    rng = np.random.default_rng(808)
    core = rng.normal(size=(300, 2))
    repeated = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])
    repeated_monitor = _monitor(window_size=60, calibration_windows=6).fit(repeated)
    repeated_result = repeated_monitor.evaluate(repeated[:60])

    assert repeated_result.ready
    assert np.isfinite(repeated_result.location_shift)
    assert np.isfinite(repeated_result.shape_shift)
    assert np.isfinite(repeated_result.max_subspace_angle)

    constant = np.ones((100, 4))
    constant_monitor = _monitor(window_size=20, calibration_windows=4).fit(constant)
    constant_result = constant_monitor.evaluate(constant[:20])
    assert constant_result.ready
    assert constant_result.location_shift == 0.0
    assert constant_result.shape_shift == 0.0
    assert constant_result.max_subspace_angle == 0.0
    assert all(value == 0.0 for value in constant_monitor.thresholds_.values())
