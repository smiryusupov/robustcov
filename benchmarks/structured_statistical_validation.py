#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Validation gate for structured covariance and PCA estimators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import robustcov as rc


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(actual - expected) / denominator)


class EmpiricalScatter:
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.location_ = X.mean(axis=0)
        self.covariance_ = np.cov(X, rowvar=False, ddof=1)
        return self


def mmcd_checks() -> dict[str, object]:
    rng = np.random.default_rng(801)
    X = rng.normal(size=(60, 3, 4))
    scale = 1e-50
    kwargs = dict(
        support_fraction=0.8,
        n_init=10,
        n_best=3,
        initial_c_steps=2,
        max_iter=12,
        flip_flop_max_iter=25,
        reweight=False,
        random_state=3,
        backend="python",
    )
    base = rc.MMCD(**kwargs).fit(X)
    tiny = rc.MMCD(**kwargs).fit(scale * X)
    covariance_error = relative_error(
        tiny.kronecker_covariance(), scale**2 * base.kronecker_covariance()
    )
    location_error = relative_error(tiny.location_, scale * base.location_)
    distance_error = relative_error(tiny.distances_, base.distances_)
    return {
        "support_equal": bool(np.array_equal(base.raw_support_, tiny.raw_support_)),
        "location_relative_error": location_error,
        "covariance_relative_error": covariance_error,
        "distance_relative_error": distance_error,
        "passed": bool(
            np.array_equal(base.raw_support_, tiny.raw_support_)
            and location_error < 1e-9
            and covariance_error < 1e-8
            and distance_error < 1e-8
        ),
    }


def kmrcd_checks() -> dict[str, object]:
    rng = np.random.default_rng(802)
    X = rng.normal(size=(64, 4))
    scale = 1e-50
    kwargs = dict(
        kernel="rbf",
        standardization="none",
        gamma="median",
        support_fraction=0.8,
        regularization=0.2,
        n_init=10,
        n_best=3,
        initial_c_steps=2,
        max_iter=20,
        random_state=2,
    )
    base = rc.KMRCD(**kwargs).fit(X)
    tiny = rc.KMRCD(**kwargs).fit(scale * X)
    indefinite_rejected = False
    try:
        rc.KMRCD(
            kernel="precomputed",
            support_fraction=1.0,
            regularization=0.2,
            n_init=1,
            n_best=1,
        ).fit(1e-10 * np.array([[1.0, 2.0], [2.0, 1.0]]))
    except ValueError:
        indefinite_rejected = True
    kernel_error = relative_error(tiny.kernel_matrix_, base.kernel_matrix_)
    distance_error = relative_error(tiny.distances_, base.distances_)
    gamma_error = abs(tiny.gamma_ * scale**2 - base.gamma_) / base.gamma_
    return {
        "support_equal": bool(np.array_equal(base.support_, tiny.support_)),
        "kernel_relative_error": kernel_error,
        "distance_relative_error": distance_error,
        "gamma_relative_error": float(gamma_error),
        "tiny_indefinite_kernel_rejected": indefinite_rejected,
        "passed": bool(
            np.array_equal(base.support_, tiny.support_)
            and kernel_error < 1e-12
            and distance_error < 1e-10
            and gamma_error < 1e-12
            and indefinite_rejected
        ),
    }


def cellmcd_checks() -> dict[str, object]:
    rng = np.random.default_rng(803)
    X = rng.normal(size=(76, 5))
    X[:5, 0] += 7.0
    X[2, 3] = np.nan
    scales = np.array([1e-50, 2.0, 0.2, 7.0, 1e20])
    offsets = np.array([1e-50, 3.0, -2.0, 9.0, -1e20])
    kwargs = dict(alpha=0.8, max_iter=35, min_samples_per_feature=None)
    base = rc.CellMCD(**kwargs).fit(X)
    transformed = rc.CellMCD(**kwargs).fit(X * scales + offsets)
    covariance_error = relative_error(
        transformed.covariance_, base.covariance_ * np.outer(scales, scales)
    )
    location_error = relative_error(
        transformed.location_, base.location_ * scales + offsets
    )
    singular_rejected = False
    singular = X.copy()
    singular[:, 2] = 1.0
    try:
        rc.CellMCD(**kwargs).fit(singular)
    except ValueError:
        singular_rejected = True
    return {
        "support_equal": bool(
            np.array_equal(base.cell_support_, transformed.cell_support_)
        ),
        "location_relative_error": location_error,
        "covariance_relative_error": covariance_error,
        "constant_feature_rejected": singular_rejected,
        "passed": bool(
            np.array_equal(base.cell_support_, transformed.cell_support_)
            and location_error < 1e-10
            and covariance_error < 1e-10
            and singular_rejected
        ),
    }


def pca_checks() -> dict[str, object]:
    rng = np.random.default_rng(804)
    X = rng.normal(size=(120, 6)) @ np.diag([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    scale = 1e-50
    base = rc.RobustPCA(n_components=3, estimator=EmpiricalScatter()).fit(X)
    tiny = rc.RobustPCA(n_components=3, estimator=EmpiricalScatter()).fit(scale * X)
    eigenvalue_error = relative_error(tiny.eigenvalues_, scale**2 * base.eigenvalues_)
    projection_error = relative_error(
        tiny.components_.T @ tiny.components_, base.components_.T @ base.components_
    )
    return {
        "eigenvalue_relative_error": eigenvalue_error,
        "projection_relative_error": projection_error,
        "passed": bool(eigenvalue_error < 1e-12 and projection_error < 1e-12),
    }


def multilinear_checks() -> dict[str, object]:
    rng = np.random.default_rng(805)
    rows, columns, ranks = 5, 6, (2, 2)
    row_components, _ = np.linalg.qr(rng.normal(size=(rows, ranks[0])))
    column_components, _ = np.linalg.qr(rng.normal(size=(columns, ranks[1])))
    cores = rng.normal(size=(46, *ranks))
    X = np.einsum(
        "au,nuv,bv->nab",
        row_components,
        cores,
        column_components,
        optimize=True,
    )
    X += 0.05 * rng.normal(size=X.shape)
    X[0, 1, 2] = np.nan
    scale = 1e-50
    kwargs = dict(ranks=ranks, max_iter=22, backend="python")
    base = rc.RobustMultilinearPCA(**kwargs).fit(X)
    tiny = rc.RobustMultilinearPCA(**kwargs).fit(scale * X)
    fitted_error = relative_error(tiny.fitted_values_, scale * base.fitted_values_)
    residual_scale_error = relative_error(
        tiny.residual_scales_, scale * base.residual_scales_
    )
    constant_rejected = False
    try:
        rc.RobustMultilinearPCA(ranks=(1, 1), max_iter=5).fit(
            np.ones((12, 3, 3))
        )
    except ValueError:
        constant_rejected = True
    return {
        "fitted_relative_error": fitted_error,
        "residual_scale_relative_error": residual_scale_error,
        "cell_mask_equal": bool(
            np.array_equal(tiny.cell_outlier_mask_, base.cell_outlier_mask_)
        ),
        "constant_sample_rejected": constant_rejected,
        "passed": bool(
            fitted_error < 1e-10
            and residual_scale_error < 1e-10
            and np.array_equal(tiny.cell_outlier_mask_, base.cell_outlier_mask_)
            and constant_rejected
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = {
        "matrix_mcd": mmcd_checks(),
        "kernel_mrcd": kmrcd_checks(),
        "cellmcd": cellmcd_checks(),
        "robust_pca": pca_checks(),
        "multilinear_pca": multilinear_checks(),
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
