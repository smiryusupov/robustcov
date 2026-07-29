#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for CellMCD and cellwise-PCA optimizations.

The script reconstructs the exact pre-optimization loop implementations,
compares complete fitted estimators, and requires at least the requested median
speedup. Timing checks are intended for a controlled local machine, not shared
CI runners.
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
import robustcov.cellmcd as cellmcd
import robustcov.cellpca as cellpca


def _legacy_weighted_scores(X, observed, weights, center, loadings, ridge):
    n = X.shape[0]
    q = loadings.shape[1]
    scores = np.zeros((n, q), dtype=np.float64)
    identity = np.eye(q)
    centered = np.where(observed, X, center) - center
    for i in range(n):
        w = weights[i]
        if not np.any(w > 0.0):
            continue
        gram = loadings.T @ (w[:, None] * loadings) + ridge * identity
        rhs = loadings.T @ (w * centered[i])
        scores[i] = np.linalg.solve(gram, rhs)
    return scores


def _legacy_weighted_loadings(
    X, observed, weights, center, scores, previous, ridge
):
    p = X.shape[1]
    q = scores.shape[1]
    identity = np.eye(q)
    centered = np.where(observed, X, center) - center
    loadings = previous.copy()
    for j in range(p):
        w = weights[:, j]
        if not np.any(w > 0.0):
            continue
        gram = scores.T @ (w[:, None] * scores) + ridge * identity
        rhs = scores.T @ (w * centered[:, j])
        loadings[j] = np.linalg.solve(gram, rhs)
    qmat, rmat = np.linalg.qr(loadings, mode="reduced")
    scores[:] = scores @ rmat.T
    return qmat


@contextmanager
def _legacy_cellpca():
    original_scores = cellpca._weighted_scores
    original_loadings = cellpca._weighted_loadings
    cellpca._weighted_scores = _legacy_weighted_scores
    cellpca._weighted_loadings = _legacy_weighted_loadings
    try:
        yield
    finally:
        cellpca._weighted_scores = original_scores
        cellpca._weighted_loadings = original_loadings


def _legacy_conditional_parameters(
    location, covariance, target, observed, observed_values
):
    if target.size == 0:
        return np.empty(0), np.empty((0, 0))
    if observed.size == 0:
        return location[target].copy(), covariance[np.ix_(target, target)].copy()
    cov_oo = covariance[np.ix_(observed, observed)]
    cov_to = covariance[np.ix_(target, observed)]
    solved = np.linalg.solve(cov_oo, observed_values - location[observed])
    mean = location[target] + cov_to @ solved
    conditional = covariance[np.ix_(target, target)] - cov_to @ np.linalg.solve(
        cov_oo, covariance[np.ix_(observed, target)]
    )
    return mean, 0.5 * (conditional + conditional.T)


def _legacy_em_update(X, W, location, covariance, minimum_eigenvalue):
    n, p = X.shape
    conditional_means = np.empty((n, p), dtype=np.float64)
    covariance_bias = np.zeros((p, p), dtype=np.float64)
    for i in range(n):
        observed = np.flatnonzero(W[i])
        missing = np.flatnonzero(~W[i])
        conditional_means[i, observed] = X[i, observed]
        mean_missing, conditional_cov = _legacy_conditional_parameters(
            location, covariance, missing, observed, X[i, observed]
        )
        conditional_means[i, missing] = mean_missing
        if missing.size:
            covariance_bias[np.ix_(missing, missing)] += conditional_cov
    new_location = conditional_means.mean(axis=0)
    centered = conditional_means - new_location
    new_covariance = (centered.T @ centered + covariance_bias) / n
    new_covariance = cellmcd._truncate_covariance(
        new_covariance, minimum_eigenvalue
    )
    return new_location, new_covariance, conditional_means


class _LegacyCellMCD(rc.CellMCD):
    def _update_cell_support(
        self, Z, W, location, covariance, penalties, finite
    ):
        column_order = np.argsort(W.sum(axis=0), kind="stable")
        for j in column_order:
            delta = np.full(Z.shape[0], np.inf, dtype=np.float64)
            other = np.arange(Z.shape[1]) != j
            patterns = np.unique(W[:, other], axis=0)
            for pattern in patterns:
                rows = np.flatnonzero(np.all(W[:, other] == pattern, axis=1))
                rows = rows[finite[rows, j]]
                if rows.size == 0:
                    continue
                observed = np.flatnonzero(other)[pattern]
                target = np.array([j], dtype=int)
                if observed.size:
                    cov_oo = covariance[np.ix_(observed, observed)]
                    cov_jo = covariance[np.ix_(target, observed)]
                    beta = np.linalg.solve(
                        cov_oo, covariance[np.ix_(observed, target)]
                    )
                    conditional_variance = float(
                        (covariance[j, j] - cov_jo @ beta).item()
                    )
                    centered = Z[np.ix_(rows, observed)] - location[observed]
                    prediction = location[j] + centered @ np.linalg.solve(
                        cov_oo, covariance[np.ix_(observed, target)]
                    ).reshape(-1)
                else:
                    conditional_variance = float(covariance[j, j])
                    prediction = np.full(rows.size, location[j], dtype=float)
                conditional_variance = max(
                    conditional_variance, self.min_eigenvalue
                )
                delta[rows] = (
                    (Z[rows, j] - prediction) ** 2 / conditional_variance
                    + np.log(conditional_variance)
                    + cellmcd._LOG_2PI
                )
            good = np.flatnonzero(delta <= penalties[j])
            if good.size < self.h_:
                good = np.argsort(delta, kind="stable")[: self.h_]
            W[:, j] = False
            W[good, j] = True
            W[~finite[:, j], j] = False
        return W

    def _diagnostics_with_support(self, X, support):
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        predictions = np.empty((n, p), dtype=float)
        conditional_std = np.empty((n, p), dtype=float)
        residuals = np.full((n, p), np.nan, dtype=float)
        for i in range(n):
            for j in range(p):
                observed = np.flatnonzero(
                    support[i] & (np.arange(p) != j)
                )
                mean, conditional = _legacy_conditional_parameters(
                    self.location_,
                    self.covariance_,
                    np.array([j], dtype=int),
                    observed,
                    X[i, observed],
                )
                variance = max(
                    float(conditional[0, 0]), np.finfo(float).tiny
                )
                predictions[i, j] = float(mean[0])
                conditional_std[i, j] = np.sqrt(variance)
                if np.isfinite(X[i, j]):
                    residuals[i, j] = (
                        X[i, j] - mean[0]
                    ) / np.sqrt(variance)
        return predictions, conditional_std, residuals

    def _partial_distances(self, X, support):
        return np.asarray(
            [
                cellmcd._partial_distance(
                    row, mask, self.location_, self.covariance_
                )
                for row, mask in zip(np.asarray(X, dtype=float), support)
            ]
        )


@contextmanager
def _legacy_cellmcd():
    original_em = cellmcd._em_update
    cellmcd._em_update = _legacy_em_update
    try:
        yield
    finally:
        cellmcd._em_update = original_em


def _timed(factory, X, context, repeats, warmups):
    for _ in range(warmups):
        with context():
            factory().fit(X)
    samples = []
    fitted = None
    for _ in range(repeats):
        with context():
            fitted = factory()
            started = perf_counter()
            fitted.fit(X)
            samples.append(perf_counter() - started)
    return median(samples), samples, fitted


def _optimized_timed(factory, X, repeats, warmups):
    for _ in range(warmups):
        factory().fit(X)
    samples = []
    fitted = None
    for _ in range(repeats):
        fitted = factory()
        started = perf_counter()
        fitted.fit(X)
        samples.append(perf_counter() - started)
    return median(samples), samples, fitted


def _cellpca_case(rng, args):
    X = rng.standard_normal((900, 45))
    X[:45, :8] += 5.0
    X[rng.random(X.shape) < 0.015] = np.nan

    def factory():
        return rc.CellwiseRobustPCA(
            n_components=6, max_iter=20, tol=1e-6
        )

    legacy_s, legacy_samples, legacy = _timed(
        factory, X, _legacy_cellpca, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _optimized_timed(
        factory, X, args.repeats, args.warmups
    )
    equivalent = bool(
        optimized.n_iter_ == legacy.n_iter_
        and np.allclose(
            optimized.center_, legacy.center_, rtol=1e-11, atol=1e-11
        )
        and np.allclose(
            optimized.components_,
            legacy.components_,
            rtol=1e-11,
            atol=1e-11,
        )
        and np.allclose(
            optimized.fitted_values_,
            legacy.fitted_values_,
            rtol=1e-11,
            atol=1e-11,
        )
        and np.allclose(
            optimized.objective_history_,
            legacy.objective_history_,
            rtol=1e-11,
            atol=1e-11,
        )
    )
    speedup = legacy_s / optimized_s
    return {
        "case": "cellwise_pca_complete_fit",
        "equivalent": equivalent,
        "legacy_median_seconds": legacy_s,
        "optimized_median_seconds": optimized_s,
        "speedup": speedup,
        "minimum_speedup": args.min_speedup,
        "passed": equivalent and speedup >= args.min_speedup,
        "legacy_samples_seconds": legacy_samples,
        "optimized_samples_seconds": optimized_samples,
    }


def _cellmcd_case(rng, args):
    X = rng.standard_normal((360, 12))
    X[:25, :4] += 6.0
    X[rng.random(X.shape) < 0.02] = np.nan
    kwargs = dict(
        alpha=0.75,
        max_iter=12,
        tol=1e-5,
        min_samples_per_feature=None,
    )
    legacy_factory = lambda: _LegacyCellMCD(**kwargs)
    optimized_factory = lambda: rc.CellMCD(**kwargs)
    legacy_s, legacy_samples, legacy = _timed(
        legacy_factory, X, _legacy_cellmcd, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _optimized_timed(
        optimized_factory, X, args.repeats, args.warmups
    )
    equivalent = bool(
        optimized.n_iter_ == legacy.n_iter_
        and np.array_equal(optimized.cell_support_, legacy.cell_support_)
        and np.allclose(
            optimized.location_, legacy.location_, rtol=1e-11, atol=1e-11
        )
        and np.allclose(
            optimized.covariance_,
            legacy.covariance_,
            rtol=1e-11,
            atol=1e-11,
        )
        and np.allclose(
            optimized.predicted_values_,
            legacy.predicted_values_,
            rtol=1e-11,
            atol=1e-11,
        )
        and np.allclose(
            optimized.distances_, legacy.distances_, rtol=1e-11, atol=1e-11
        )
    )
    speedup = legacy_s / optimized_s
    return {
        "case": "cellmcd_complete_fit",
        "equivalent": equivalent,
        "legacy_median_seconds": legacy_s,
        "optimized_median_seconds": optimized_s,
        "speedup": speedup,
        "minimum_speedup": args.min_speedup,
        "passed": equivalent and speedup >= args.min_speedup,
        "legacy_samples_seconds": legacy_samples,
        "optimized_samples_seconds": optimized_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("all", "cellpca", "cellmcd"), default="all"
    )
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.min_speedup <= 1.0:
        parser.error("--min-speedup must be greater than 1")
    if args.repeats < 3 or args.warmups < 1:
        parser.error("use at least 3 repeats and 1 warmup")

    rng = np.random.default_rng(args.seed)
    cases = {"cellpca": _cellpca_case, "cellmcd": _cellmcd_case}
    selected = list(cases) if args.case == "all" else [args.case]
    results = [cases[name](rng, args) for name in selected]
    report = {
        "minimum_speedup": args.min_speedup,
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
