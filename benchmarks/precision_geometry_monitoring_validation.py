#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Validation gate for sparse precision, feature geometry, and monitoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(actual - expected) / denominator)


def spd_sample(seed=0, n=160, p=5):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(p, p))
    covariance = matrix @ matrix.T + np.eye(p)
    location = rng.normal(size=p)
    X = rng.multivariate_normal(location, covariance, size=n)
    return X, location, covariance


def sparse_precision_checks() -> dict[str, object]:
    X, location, covariance = spd_sample(seed=901)
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
    tiny = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(1e-100 * covariance, 1e-50 * location),
        standardize=True,
    ).fit(1e-50 * X)
    base_unpenalized = rc.RobustGraphicalLasso(
        alpha=0.0,
        scatter_estimator=FixedScatter(covariance, location),
        standardize=True,
    ).fit(X)

    rng = np.random.default_rng(902)
    core = rng.normal(size=(140, 3))
    repeated = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])
    singular = rc.RobustGraphicalLasso(
        alpha=0.05,
        scatter_estimator="empirical",
        standardize=True,
        max_iter=1000,
    ).fit(repeated)
    invalid_rejected = False
    try:
        rc.RobustGraphicalLasso(
            alpha=0.0,
            scatter_estimator=FixedScatter([[1.0, 2.0], [2.0, 1.0]]),
        ).fit(np.zeros((10, 2)))
    except ValueError:
        invalid_rejected = True

    precision_error = relative_error(
        transformed.precision_ * np.outer(scales, scales), base.precision_
    )
    partial_error = relative_error(
        transformed.partial_correlation_, base.partial_correlation_
    )
    distance_error = relative_error(
        transformed.mahalanobis(X[:20] * scales), base.mahalanobis(X[:20])
    )
    tiny_distance_error = relative_error(
        tiny.mahalanobis(1e-50 * X[:20]), base_unpenalized.mahalanobis(X[:20])
    )
    inverse_error = relative_error(
        singular.covariance_ @ singular.precision_, np.eye(repeated.shape[1])
    )
    passed = bool(
        precision_error < 1e-10
        and partial_error < 1e-10
        and distance_error < 1e-10
        and tiny_distance_error < 1e-10
        and np.array_equal(transformed.adjacency_, base.adjacency_)
        and inverse_error < 1e-6
        and invalid_rejected
    )
    return {
        "feature_scale_precision_relative_error": precision_error,
        "feature_scale_partial_relative_error": partial_error,
        "feature_scale_distance_relative_error": distance_error,
        "tiny_unit_distance_relative_error": tiny_distance_error,
        "adjacency_equal": bool(np.array_equal(transformed.adjacency_, base.adjacency_)),
        "singular_inverse_relative_error": inverse_error,
        "partial_constant_feature_detected": bool(singular.constant_features_[-1]),
        "indefinite_scatter_rejected": invalid_rejected,
        "passed": passed,
    }


def spatial_sign_checks() -> dict[str, object]:
    rng = np.random.default_rng(903)
    half = rng.normal(size=(90, 6))
    shift = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
    X = np.vstack([half, -half]) + shift
    permutation = np.array([4, 1, 5, 0, 3, 2])
    base = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(X)
    tiny = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(1e-200 * X)
    permuted = rc.SGLASSO(alpha=0.08, max_iter=1000).fit(X[:, permutation])
    coincident_rejected = False
    try:
        rc.SGLASSO(alpha=0.0).fit(np.ones((30, 5)))
    except ValueError:
        coincident_rejected = True

    scale_precision_error = relative_error(tiny.precision_, base.precision_)
    scale_location_error = relative_error(tiny.location_, 1e-200 * base.location_)
    permutation_error = relative_error(
        permuted.precision_, base.precision_[np.ix_(permutation, permutation)]
    )
    passed = bool(
        scale_precision_error < 1e-9
        and scale_location_error < 1e-9
        and permutation_error < 1e-9
        and np.array_equal(tiny.adjacency_, base.adjacency_)
        and coincident_rejected
    )
    return {
        "tiny_scale_precision_relative_error": scale_precision_error,
        "tiny_scale_location_relative_error": scale_location_error,
        "permutation_precision_relative_error": permutation_error,
        "adjacency_equal": bool(np.array_equal(tiny.adjacency_, base.adjacency_)),
        "coincident_sample_rejected": coincident_rejected,
        "passed": passed,
    }


def feature_geometry_checks() -> dict[str, object]:
    X, location, covariance = spd_sample(seed=904)
    permutation = np.array([3, 0, 4, 1, 2])
    base = rc.FeatureGeometry(
        estimator=FixedScatter(covariance, location)
    ).fit(X)
    tiny = rc.FeatureGeometry(
        estimator=FixedScatter(1e-100 * covariance, 1e-50 * location)
    ).fit(1e-50 * X)
    permuted = rc.FeatureGeometry(
        estimator=FixedScatter(
            covariance[np.ix_(permutation, permutation)], location[permutation]
        )
    ).fit(X[:, permutation])

    rng = np.random.default_rng(905)
    core = rng.normal(size=(120, 2))
    repeated = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])
    singular = rc.FeatureGeometry(estimator=EmpiricalScatter()).fit(repeated)
    invalid_rejected = False
    constant_rejected = False
    try:
        rc.FeatureGeometry(
            estimator=FixedScatter([[1.0, 2.0], [2.0, 1.0]])
        ).fit(np.zeros((10, 2)))
    except ValueError:
        invalid_rejected = True
    try:
        rc.FeatureGeometry(estimator=EmpiricalScatter()).fit(np.ones((20, 3)))
    except ValueError:
        constant_rejected = True

    scale_distance_error = relative_error(
        tiny.squared_mahalanobis(1e-50 * X[:20]),
        base.squared_mahalanobis(X[:20]),
    )
    permutation_distance_error = relative_error(
        permuted.squared_mahalanobis(X[:20, permutation]),
        base.squared_mahalanobis(X[:20]),
    )
    inverse_error = relative_error(
        singular.covariance_ @ singular.precision_, np.eye(repeated.shape[1])
    )
    passed = bool(
        scale_distance_error < 1e-10
        and permutation_distance_error < 1e-10
        and inverse_error < 1e-5
        and invalid_rejected
        and constant_rejected
    )
    return {
        "tiny_scale_distance_relative_error": scale_distance_error,
        "permutation_distance_relative_error": permutation_distance_error,
        "singular_inverse_relative_error": inverse_error,
        "indefinite_covariance_rejected": invalid_rejected,
        "constant_covariance_rejected": constant_rejected,
        "passed": passed,
    }


def monitoring_checks() -> dict[str, object]:
    rng = np.random.default_rng(906)
    latent = rng.normal(size=(420, 2))
    basis = np.zeros((5, 2))
    basis[0, 0] = 2.5
    basis[1, 1] = 1.5
    basis[2, :] = [0.6, -0.3]
    X = latent @ basis.T + rng.normal(scale=0.08, size=(420, 5))
    batch = X[:80].copy()
    batch[:, 0] += 0.5
    permutation = np.array([3, 1, 4, 0, 2])

    kwargs = dict(
        n_components=2,
        estimator=RelativeEmpiricalScatter(),
        window_size=80,
        calibration_windows=8,
        threshold_quantile=0.99,
        sample_quantile=0.98,
        random_state=0,
    )
    base_monitor = rc.RobustSubspaceMonitor(**kwargs).fit(X)
    base = base_monitor.evaluate(batch)
    tiny_monitor = rc.RobustSubspaceMonitor(**kwargs).fit(1e-50 * X)
    tiny = tiny_monitor.evaluate(1e-50 * batch)
    permuted_monitor = rc.RobustSubspaceMonitor(**kwargs).fit(X[:, permutation])
    permuted = permuted_monitor.evaluate(batch[:, permutation])

    metric_names = (
        "location_shift",
        "scale_shift",
        "shape_shift",
        "max_subspace_angle",
        "score_distance_shift",
        "orthogonal_distance_shift",
        "combined_outlier_fraction",
    )
    scale_metric_error = max(
        abs(float(getattr(tiny, name)) - float(getattr(base, name)))
        for name in metric_names
    )
    permutation_metric_error = max(
        abs(float(getattr(permuted, name)) - float(getattr(base, name)))
        for name in metric_names
    )
    threshold_scale_error = max(
        abs(tiny_monitor.thresholds_[name] - base_monitor.thresholds_[name])
        for name in base_monitor.thresholds_
    )
    threshold_permutation_error = max(
        abs(permuted_monitor.thresholds_[name] - base_monitor.thresholds_[name])
        for name in base_monitor.thresholds_
    )

    core = rng.normal(size=(300, 2))
    repeated = np.column_stack([core, core[:, 0], np.ones(core.shape[0])])
    repeated_monitor = rc.RobustSubspaceMonitor(
        **{**kwargs, "window_size": 60, "calibration_windows": 6}
    ).fit(repeated)
    repeated_ready = repeated_monitor.evaluate(repeated[:60]).ready
    constant = np.ones((100, 4))
    constant_monitor = rc.RobustSubspaceMonitor(
        **{**kwargs, "window_size": 20, "calibration_windows": 4}
    ).fit(constant)
    constant_result = constant_monitor.evaluate(constant[:20])
    constant_zero = bool(
        constant_result.location_shift == 0.0
        and constant_result.shape_shift == 0.0
        and constant_result.max_subspace_angle == 0.0
        and all(value == 0.0 for value in constant_monitor.thresholds_.values())
    )
    passed = bool(
        scale_metric_error < 1e-9
        and permutation_metric_error < 1e-9
        and threshold_scale_error < 1e-9
        and threshold_permutation_error < 1e-9
        and repeated_ready
        and constant_zero
    )
    return {
        "tiny_scale_max_metric_absolute_error": scale_metric_error,
        "permutation_max_metric_absolute_error": permutation_metric_error,
        "tiny_scale_max_threshold_absolute_error": threshold_scale_error,
        "permutation_max_threshold_absolute_error": threshold_permutation_error,
        "repeated_feature_monitor_ready": bool(repeated_ready),
        "constant_reference_is_stable": constant_zero,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = {
        "robust_graphical_lasso": sparse_precision_checks(),
        "spatial_sign_graphical_lasso": spatial_sign_checks(),
        "feature_geometry": feature_geometry_checks(),
        "robust_subspace_monitor": monitoring_checks(),
    }
    report["passed"] = bool(all(section["passed"] for section in report.values()))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
