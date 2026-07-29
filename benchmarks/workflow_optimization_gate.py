#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for sparse-PCA and M-scatter workflow optimizations.

The gate reconstructs the exact pre-optimization helper implementations,
compares complete fitted outputs, and requires the requested median speedup.
Timing checks are intended for a controlled local machine, not shared CI.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import robustcov as rc
import robustcov.m_estimators as m_estimators
import robustcov.sparse_cellpca as sparse_cellpca


def _legacy_soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _legacy_weighted_elastic_net_loadings(
    X,
    observed,
    weights,
    center,
    scores,
    previous,
    alpha,
    l1_ratio,
    ridge,
    max_iter,
    tol,
):
    _, p = X.shape
    q = scores.shape[1]
    safe = np.where(observed, X, center)
    centered = safe - center
    loadings = previous.copy()
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    for j in range(p):
        w = np.asarray(weights[:, j], dtype=float)
        active = w > 0.0
        if int(np.count_nonzero(active)) < 2:
            continue

        design = scores[active]
        response = centered[active, j]
        active_weights = w[active]
        weight_sum = max(
            float(np.sum(active_weights)), np.sqrt(sparse_cellpca._EPS)
        )
        beta = loadings[j].copy()
        residual = response - design @ beta

        for _ in range(max_iter):
            old = beta.copy()
            for k in range(q):
                column = design[:, k]
                residual += column * beta[k]
                curvature = float(
                    np.dot(active_weights, column * column) / weight_sum
                )
                curvature += float(l2[k]) + ridge
                correlation = float(
                    np.dot(active_weights, column * residual) / weight_sum
                )
                beta[k] = _legacy_soft_threshold(
                    correlation, float(l1[k])
                ) / max(curvature, np.sqrt(sparse_cellpca._EPS))
                residual -= column * beta[k]
            if np.linalg.norm(beta - old) <= tol * max(
                np.linalg.norm(old), 1.0
            ):
                break
        loadings[j] = beta
    return loadings


def _legacy_safe_pinv(matrix):
    return np.linalg.pinv(0.5 * (matrix + matrix.T), hermitian=True)


def _legacy_mahalanobis(centered, precision):
    return np.einsum(
        "ij,jk,ik->i", centered, precision, centered, optimize=True
    )


@contextmanager
def _legacy_sparse_context():
    original = sparse_cellpca._weighted_elastic_net_loadings
    sparse_cellpca._weighted_elastic_net_loadings = (
        _legacy_weighted_elastic_net_loadings
    )
    try:
        yield
    finally:
        sparse_cellpca._weighted_elastic_net_loadings = original


@contextmanager
def _legacy_m_scatter_context():
    original_inverse = m_estimators._safe_pinv
    original_distance = m_estimators._mahalanobis_from_precision
    m_estimators._safe_pinv = _legacy_safe_pinv
    m_estimators._mahalanobis_from_precision = _legacy_mahalanobis
    try:
        yield
    finally:
        m_estimators._safe_pinv = original_inverse
        m_estimators._mahalanobis_from_precision = original_distance


@contextmanager
def _optimized_context():
    yield


def _timed(factory, context, repeats, warmups):
    for _ in range(warmups):
        with context():
            factory()
    samples = []
    fitted = None
    for _ in range(repeats):
        with context():
            started = perf_counter()
            fitted = factory()
            samples.append(perf_counter() - started)
    return median(samples), samples, fitted


def _result(
    name,
    legacy_seconds,
    optimized_seconds,
    legacy_samples,
    optimized_samples,
    equivalent,
    minimum_speedup,
):
    speedup = legacy_seconds / optimized_seconds
    return {
        "case": name,
        "equivalent": bool(equivalent),
        "legacy_median_seconds": legacy_seconds,
        "optimized_median_seconds": optimized_seconds,
        "speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "passed": bool(equivalent and speedup >= minimum_speedup),
        "legacy_samples_seconds": legacy_samples,
        "optimized_samples_seconds": optimized_samples,
    }


def _sparse_case(rng, args):
    X = rng.standard_normal((600, 160))
    X[:40, :20] += 4.0
    X[rng.random(X.shape) < 0.015] = np.nan
    kwargs = dict(
        n_components=8,
        max_iter=10,
        tol=1e-6,
        alpha=0.04,
        loading_max_iter=80,
        loading_tol=1e-7,
    )

    def factory():
        return rc.SparseCellPCA(**kwargs).fit(X)

    legacy_s, legacy_samples, legacy = _timed(
        factory, _legacy_sparse_context, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _timed(
        factory, _optimized_context, args.repeats, args.warmups
    )
    equivalent = (
        optimized.n_iter_ == legacy.n_iter_
        and np.array_equal(
            optimized.loading_support_, legacy.loading_support_
        )
        and np.allclose(
            optimized.center_, legacy.center_, rtol=1e-10, atol=1e-11
        )
        and np.allclose(
            optimized.components_, legacy.components_, rtol=1e-10, atol=1e-11
        )
        and np.allclose(
            optimized.fitted_values_,
            legacy.fitted_values_,
            rtol=1e-10,
            atol=1e-11,
        )
        and np.allclose(
            optimized.objective_history_,
            legacy.objective_history_,
            rtol=1e-10,
            atol=1e-9,
        )
    )
    return _result(
        "sparse_cellpca_complete_fit",
        legacy_s,
        optimized_s,
        legacy_samples,
        optimized_samples,
        equivalent,
        args.min_speedup,
    )


def _auto_scatter_case(rng, args):
    X = rng.standard_t(df=3, size=(700, 24))
    X[:50, :6] += 4.0

    def factory():
        return rc.AutoRobustScatter(
            selection="stability",
            n_splits=2,
            subsample_fraction=0.7,
            random_state=7,
        ).fit(X)

    legacy_s, legacy_samples, legacy = _timed(
        factory, _legacy_m_scatter_context, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _timed(
        factory, _optimized_context, args.repeats, args.warmups
    )
    legacy_scores = np.asarray(
        [result.score for result in legacy.candidate_results_]
    )
    optimized_scores = np.asarray(
        [result.score for result in optimized.candidate_results_]
    )
    equivalent = (
        optimized.best_estimator_name_ == legacy.best_estimator_name_
        and optimized.n_iter_ == legacy.n_iter_
        and np.allclose(
            optimized.location_, legacy.location_, rtol=1e-10, atol=1e-11
        )
        and np.allclose(
            optimized.covariance_,
            legacy.covariance_,
            rtol=1e-10,
            atol=1e-11,
        )
        and np.allclose(
            optimized.distances_, legacy.distances_, rtol=1e-10, atol=1e-9
        )
        and np.allclose(
            optimized_scores, legacy_scores, rtol=1e-10, atol=1e-11
        )
    )
    return _result(
        "auto_scatter_complete_fit",
        legacy_s,
        optimized_s,
        legacy_samples,
        optimized_samples,
        equivalent,
        args.min_speedup,
    )


def _monitor_case(rng, args):
    X = rng.standard_normal((1200, 26))
    X[:50, :5] += 2.5

    def factory():
        return rc.RobustSubspaceMonitor(
            n_components=6,
            window_size=180,
            calibration_windows=8,
            random_state=7,
        ).fit(X)

    legacy_s, legacy_samples, legacy = _timed(
        factory, _legacy_m_scatter_context, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _timed(
        factory, _optimized_context, args.repeats, args.warmups
    )
    legacy_thresholds = np.asarray(
        [legacy.thresholds_[name] for name in sorted(legacy.thresholds_)]
    )
    optimized_thresholds = np.asarray(
        [optimized.thresholds_[name] for name in sorted(optimized.thresholds_)]
    )
    equivalent = (
        optimized.n_components_ == legacy.n_components_
        and np.allclose(
            optimized.reference_location_,
            legacy.reference_location_,
            rtol=1e-10,
            atol=1e-11,
        )
        and np.allclose(
            optimized.reference_covariance_,
            legacy.reference_covariance_,
            rtol=1e-10,
            atol=1e-11,
        )
        and np.allclose(
            optimized.reference_components_,
            legacy.reference_components_,
            rtol=1e-9,
            atol=1e-10,
        )
        and np.allclose(
            optimized_thresholds,
            legacy_thresholds,
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True,
        )
    )
    return _result(
        "subspace_monitor_complete_fit",
        legacy_s,
        optimized_s,
        legacy_samples,
        optimized_samples,
        equivalent,
        args.min_speedup,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "sparse-cellpca", "auto-scatter", "monitor"),
        default="all",
    )
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.min_speedup <= 0:
        parser.error("--min-speedup must be positive")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")

    selected = {
        "sparse-cellpca": _sparse_case,
        "auto-scatter": _auto_scatter_case,
        "monitor": _monitor_case,
    }
    if args.case != "all":
        selected = {args.case: selected[args.case]}

    results = []
    for offset, (name, run) in enumerate(selected.items()):
        result = run(np.random.default_rng(args.seed + offset), args)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status} {name}: {result['speedup']:.2f}x "
            f"({result['legacy_median_seconds']:.6f}s -> "
            f"{result['optimized_median_seconds']:.6f}s), "
            f"equivalent={result['equivalent']}"
        )

    payload = {
        "minimum_speedup": args.min_speedup,
        "threads": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
