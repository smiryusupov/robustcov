#!/usr/bin/env python
"""OpenMP scaling benchmark for every currently threaded native workload.

The benchmark separates complete estimators from low-level kernels.  Native
algorithms that are not OpenMP-parallel (for example the C++ joint diagonalizer
used by SOBI) belong in their dedicated native-backend gate and are not included
here merely because they are implemented in C++.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import robustcov as rc
from robustcov._native import require_native


OPENMP_WORKLOAD_KEYS = (
    "fastmcd",
    "tyler",
    "regularized_tyler",
    "vector_mahalanobis",
    "matrix_mahalanobis",
    "weighted_tucker",
)


@dataclass(frozen=True)
class ScalingWorkload:
    key: str
    label: str
    scope: str
    run: Callable[[int], np.ndarray]


def timed(fn: Callable[[], np.ndarray], repeat: int):
    values = []
    output = None
    # Warm-up avoids charging one-time allocation/import effects to a thread count.
    output = fn()
    for _ in range(repeat):
        start = time.perf_counter()
        output = fn()
        values.append(time.perf_counter() - start)
    arr = np.asarray(values)
    return float(np.median(arr)), float(arr.min()), float(arr.max()), np.asarray(output)


def build_workloads(n: int, p: int, seed: int) -> list[ScalingWorkload]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[: max(1, n // 10)] += 8.0
    cpp = require_native("OpenMP scaling benchmark")

    location = np.mean(X, axis=0)
    covariance = np.cov(X, rowvar=False) + 0.1 * np.eye(p)
    precision = np.linalg.inv(covariance)

    matrix_n = max(512, n // 2)
    n_rows = max(8, min(16, p))
    n_columns = max(7, min(14, p - 1 if p > 8 else p))
    X_matrix = rng.normal(size=(matrix_n, n_rows, n_columns))
    matrix_location = np.mean(X_matrix, axis=0)
    row_precision = np.linalg.inv(np.cov(X_matrix.mean(axis=2), rowvar=False) + 0.1 * np.eye(n_rows))
    column_precision = np.linalg.inv(np.cov(X_matrix.mean(axis=1), rowvar=False) + 0.1 * np.eye(n_columns))

    tucker_n = max(256, n // 4)
    X_tucker = X_matrix[:tucker_n]
    weights = rng.uniform(0.2, 1.0, size=X_tucker.shape)
    q_rows = min(4, n_rows)
    q_columns = min(3, n_columns)
    row_components, _ = np.linalg.qr(rng.normal(size=(n_rows, q_rows)))
    column_components, _ = np.linalg.qr(rng.normal(size=(n_columns, q_columns)))

    def fast_mcd(threads: int) -> np.ndarray:
        fitted = rc.FastMCD(
            n_init=220,
            n_best=8,
            quality="fast",
            random_state=0,
            n_jobs=threads,
        ).fit(X)
        return fitted.covariance_

    def tyler_shape(threads: int) -> np.ndarray:
        return rc.TylerShape(
            max_iter=100,
            tol=1e-7,
            scale_correction="radial_median",
            n_jobs=threads,
        ).fit(X).covariance_

    def regularized_tyler(threads: int) -> np.ndarray:
        return rc.RegularizedTyler(
            alpha=0.1,
            max_iter=100,
            tol=1e-7,
            scale_correction="radial_median",
            n_jobs=threads,
        ).fit(X).covariance_

    def vector_mahalanobis(threads: int) -> np.ndarray:
        rc.set_num_threads(threads)
        return np.asarray(cpp.mahalanobis2_batch(X, location, precision))

    def matrix_mahalanobis(threads: int) -> np.ndarray:
        rc.set_num_threads(threads)
        return np.asarray(
            cpp.matrix_mahalanobis2_batch(
                X_matrix, matrix_location, row_precision, column_precision
            )
        )

    def weighted_tucker(threads: int) -> np.ndarray:
        rc.set_num_threads(threads)
        return np.asarray(
            cpp.weighted_tucker_scores_2d(
                X_tucker,
                weights,
                matrix_location,
                row_components,
                column_components,
                1e-8,
            )
        )

    return [
        ScalingWorkload("fastmcd", "FastMCD complete fit", "estimator", fast_mcd),
        ScalingWorkload("tyler", "TylerShape complete fit", "estimator", tyler_shape),
        ScalingWorkload(
            "regularized_tyler",
            "RegularizedTyler complete fit",
            "estimator",
            regularized_tyler,
        ),
        ScalingWorkload(
            "vector_mahalanobis",
            "Vector Mahalanobis batch",
            "native kernel",
            vector_mahalanobis,
        ),
        ScalingWorkload(
            "matrix_mahalanobis",
            "Matrix Mahalanobis batch",
            "native kernel",
            matrix_mahalanobis,
        ),
        ScalingWorkload(
            "weighted_tucker",
            "Weighted Tucker score solve",
            "native kernel",
            weighted_tucker,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=OPENMP_WORKLOAD_KEYS,
        default=None,
    )
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if 1 not in args.threads:
        parser.error("--threads must include 1 so speedup and numerical drift have a baseline")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if not rc.has_openmp():
        raise RuntimeError("the installed robustcov native extension was built without OpenMP")

    workloads = build_workloads(args.n, args.p, seed=0)
    if args.workloads:
        keys = set(args.workloads)
        workloads = [workload for workload in workloads if workload.key in keys]

    original_threads = rc.get_num_threads()
    rows = []
    try:
        print(f"openmp_enabled={rc.has_openmp()}, default_threads={original_threads}")
        for workload in workloads:
            baseline_time = None
            baseline_output = None
            for threads in args.threads:
                med, mn, mx, output = timed(
                    lambda workload=workload, threads=threads: workload.run(threads),
                    args.repeat,
                )
                if threads == 1:
                    baseline_time = med
                    baseline_output = output.copy()
                assert baseline_time is not None and baseline_output is not None
                speedup = baseline_time / med if med > 0 else float("nan")
                diff = float(np.max(np.abs(output - baseline_output)))
                scale = max(float(np.max(np.abs(baseline_output))), np.finfo(float).tiny)
                relative_diff = diff / scale
                row = {
                    "workload": workload.key,
                    "method": workload.label,
                    "scope": workload.scope,
                    "threads": threads,
                    "median_seconds": med,
                    "min_seconds": mn,
                    "max_seconds": mx,
                    "speedup_vs_1": speedup,
                    "max_abs_diff_vs_1": diff,
                    "max_relative_diff_vs_1": relative_diff,
                }
                rows.append(row)
                print(
                    f"{workload.key},{threads},{med:.6f},{speedup:.3f},"
                    f"abs_diff={diff:.3e},rel_diff={relative_diff:.3e}"
                )
    finally:
        rc.set_num_threads(original_threads)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
