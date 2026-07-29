#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Acceptance gate for Python-to-C++ numerical kernel ports.

A native kernel is accepted only when it is numerically equivalent to its NumPy
reference and reaches the requested median speedup on a representative workload.
This is intentionally a manual/release benchmark rather than a unit test because
shared CI machines are too noisy for stable timing assertions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

# Keep NumPy's BLAS baseline deterministic and avoid accidental oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from robustcov import get_num_threads, set_num_threads
from robustcov._native import (
    mahalanobis_squared_batch,
    matrix_mahalanobis2_batch,
    native_available,
    weighted_tucker_scores_2d,
)


def _timed(callable_, repeats: int, warmups: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        callable_()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        callable_()
        samples.append(perf_counter() - start)
    return median(samples), samples


def _evaluate(name, python_call, cpp_call, *, repeats, warmups, min_speedup, rtol, atol):
    expected = python_call()
    actual = cpp_call()
    equivalent = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    max_abs_error = float(np.max(np.abs(actual - expected)))
    python_s, python_samples = _timed(python_call, repeats, warmups)
    cpp_s, cpp_samples = _timed(cpp_call, repeats, warmups)
    speedup = python_s / cpp_s if cpp_s > 0 else float("inf")
    return {
        "kernel": name,
        "equivalent": equivalent,
        "max_abs_error": max_abs_error,
        "python_median_seconds": python_s,
        "cpp_median_seconds": cpp_s,
        "speedup": speedup,
        "minimum_speedup": min_speedup,
        "passed": equivalent and speedup >= min_speedup,
        "python_samples_seconds": python_samples,
        "cpp_samples_seconds": cpp_samples,
    }


def _vector_case(rng, args):
    n, p = 50_000, 16
    X = rng.normal(size=(n, p))
    location = rng.normal(size=p)
    A = rng.normal(size=(p, p))
    precision = np.asarray(A @ A.T + np.eye(p), order="C")
    return _evaluate(
        "mahalanobis_squared_batch",
        lambda: mahalanobis_squared_batch(X, location, precision, backend="python"),
        lambda: mahalanobis_squared_batch(X, location, precision, backend="cpp"),
        repeats=args.repeats,
        warmups=args.warmups,
        min_speedup=args.min_speedup,
        rtol=1e-12,
        atol=1e-9,
    )


def _matrix_case(rng, args):
    # This is the workload where auto-routing enables the native path.
    n, r, c = 20_000, 8, 10
    X = rng.normal(size=(n, r, c))
    location = rng.normal(size=(r, c))
    A = rng.normal(size=(r, r))
    B = rng.normal(size=(c, c))
    row_precision = np.asarray(A @ A.T + np.eye(r), order="C")
    column_precision = np.asarray(B @ B.T + np.eye(c), order="C")
    return _evaluate(
        "matrix_mahalanobis2_batch",
        lambda: matrix_mahalanobis2_batch(
            X, location, row_precision, column_precision, backend="python"
        ),
        lambda: matrix_mahalanobis2_batch(
            X, location, row_precision, column_precision, backend="cpp"
        ),
        repeats=args.repeats,
        warmups=args.warmups,
        min_speedup=args.min_speedup,
        rtol=1e-12,
        atol=1e-8,
    )


def _tucker_case(rng, args):
    n, r, c, q1, q2 = 128, 10, 12, 3, 3
    X = rng.normal(size=(n, r, c))
    weights = rng.uniform(size=X.shape)
    weights[rng.random(X.shape) < 0.15] = 0.0
    center = rng.normal(size=(r, c))
    row_components, _ = np.linalg.qr(rng.normal(size=(r, q1)))
    column_components, _ = np.linalg.qr(rng.normal(size=(c, q2)))
    return _evaluate(
        "weighted_tucker_scores_2d",
        lambda: weighted_tucker_scores_2d(
            X,
            weights,
            center,
            row_components,
            column_components,
            ridge=1e-7,
            backend="python",
        ),
        lambda: weighted_tucker_scores_2d(
            X,
            weights,
            center,
            row_components,
            column_components,
            ridge=1e-7,
            backend="cpp",
        ),
        repeats=args.repeats,
        warmups=args.warmups,
        min_speedup=args.min_speedup,
        rtol=1e-11,
        atol=1e-10,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("all", "vector-mahalanobis", "matrix-mahalanobis", "weighted-tucker"),
        default="all",
    )
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.min_speedup <= 1.0:
        parser.error("--min-speedup must be greater than 1")
    if args.repeats < 3 or args.warmups < 1 or args.threads < 1:
        parser.error("use at least 3 repeats, 1 warmup, and 1 thread")
    if not native_available():
        raise SystemExit("robustcov native extension is unavailable")

    previous_threads = get_num_threads()
    set_num_threads(args.threads)
    try:
        rng = np.random.default_rng(args.seed)
        cases = {
            "vector-mahalanobis": _vector_case,
            "matrix-mahalanobis": _matrix_case,
            "weighted-tucker": _tucker_case,
        }
        selected = list(cases) if args.kernel == "all" else [args.kernel]
        results = [cases[name](rng, args) for name in selected]
    finally:
        set_num_threads(previous_threads)

    report = {
        "minimum_speedup": args.min_speedup,
        "threads": args.threads,
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
