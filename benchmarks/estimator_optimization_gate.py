#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for profiling-driven estimator optimizations.

The gate compares the current implementation with the exact pre-optimization
algorithm on representative complete fits. It verifies fitted-result
equivalence and requires the requested median speedup. This is a local/manual
benchmark because timing assertions are not stable on shared CI runners.
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
import robustcov.mmcd as mmcd
from robustcov._native import resolve_backend


class _LegacyNumpyProxy:
    """Forward NumPy calls while restoring the two pre-optimization paths."""

    def __init__(self, module, *, vector: bool = False, matrix: bool = False):
        self._module = module
        self._vector = vector
        self._matrix = matrix

    def __getattr__(self, name):
        return getattr(self._module, name)

    def einsum(self, subscripts, *operands, **kwargs):
        if self._vector and subscripts == "ij,jk,ik->i":
            kwargs.pop("optimize", None)
            return self._module.einsum(subscripts, *operands, **kwargs)
        if self._matrix and subscripts == "nrc,cd,nsd->rs":
            residuals, precision, _ = operands
            output = self._module.zeros(
                (residuals.shape[1], residuals.shape[1]), dtype=self._module.float64
            )
            for residual in residuals:
                output += residual @ precision @ residual.T
            return output
        if self._matrix and subscripts == "nrc,rs,nsd->cd":
            residuals, precision, _ = operands
            output = self._module.zeros(
                (residuals.shape[2], residuals.shape[2]), dtype=self._module.float64
            )
            for residual in residuals:
                output += residual.T @ precision @ residual
            return output
        return self._module.einsum(subscripts, *operands, **kwargs)


def _legacy_mahalanobis(centered, precision):
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


@contextmanager
def _legacy_m_scatter():
    original = m_estimators._mahalanobis_from_precision
    m_estimators._mahalanobis_from_precision = _legacy_mahalanobis
    try:
        yield
    finally:
        m_estimators._mahalanobis_from_precision = original


class _LegacyMatrixMCD(rc.MatrixMCD):
    """Matrix MCD with the previous eager auto-to-C++ resolution."""

    def _validate_parameters(self, *, resolve: bool = True) -> None:
        super()._validate_parameters(resolve=resolve)
        if resolve and self.backend == "auto":
            self.backend_ = resolve_backend("auto")


@contextmanager
def _legacy_matrix_mcd():
    original = mmcd.np
    mmcd.np = _LegacyNumpyProxy(np, matrix=True)
    try:
        yield
    finally:
        mmcd.np = original


def _timed(factory, X, context, repeats: int, warmups: int):
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


def _optimized_timed(factory, X, repeats: int, warmups: int):
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


def _m_scatter_case(rng, args):
    X = rng.standard_normal((4_000, 64))
    X[:200, :10] += 4.0

    def factory():
        return rc.RegularizedCauchy(
            alpha=0.1,
            max_iter=150,
            tol=1e-7,
            warn_on_nonconvergence=False,
        )

    legacy_s, legacy_samples, legacy = _timed(
        factory, X, _legacy_m_scatter, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _optimized_timed(
        factory, X, args.repeats, args.warmups
    )
    equivalent = bool(
        np.allclose(optimized.location_, legacy.location_, rtol=1e-12, atol=1e-12)
        and np.allclose(
            optimized.covariance_, legacy.covariance_, rtol=1e-11, atol=1e-11
        )
        and optimized.n_iter_ == legacy.n_iter_
    )
    speedup = legacy_s / optimized_s
    return {
        "case": "regularized_cauchy_complete_fit",
        "equivalent": equivalent,
        "legacy_median_seconds": legacy_s,
        "optimized_median_seconds": optimized_s,
        "speedup": speedup,
        "minimum_speedup": args.min_speedup,
        "passed": equivalent and speedup >= args.min_speedup,
        "legacy_samples_seconds": legacy_samples,
        "optimized_samples_seconds": optimized_samples,
    }


def _matrix_mcd_case(rng, args):
    X = rng.standard_normal((350, 10, 10))
    X[:32, :4, :4] += 4.0
    kwargs = dict(
        n_init=20,
        n_best=4,
        initial_c_steps=2,
        max_iter=20,
        flip_flop_initial_iter=2,
        flip_flop_max_iter=30,
        random_state=7,
        backend="auto",
    )
    legacy_factory = lambda: _LegacyMatrixMCD(**kwargs)
    optimized_factory = lambda: rc.MatrixMCD(**kwargs)
    legacy_s, legacy_samples, legacy = _timed(
        legacy_factory, X, _legacy_matrix_mcd, args.repeats, args.warmups
    )
    optimized_s, optimized_samples, optimized = _optimized_timed(
        optimized_factory, X, args.repeats, args.warmups
    )
    equivalent = bool(
        np.array_equal(optimized.support_, legacy.support_)
        and np.allclose(optimized.location_, legacy.location_, rtol=1e-11, atol=1e-11)
        and np.allclose(
            optimized.row_covariance_, legacy.row_covariance_, rtol=1e-10, atol=1e-10
        )
        and np.allclose(
            optimized.column_covariance_,
            legacy.column_covariance_,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    speedup = legacy_s / optimized_s
    return {
        "case": "matrix_mcd_complete_fit",
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
        "--case", choices=("all", "m-scatter", "matrix-mcd"), default="all"
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
    cases = {"m-scatter": _m_scatter_case, "matrix-mcd": _matrix_mcd_case}
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
