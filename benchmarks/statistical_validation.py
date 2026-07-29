#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Reproducible statistical validation for robust covariance estimators.

The checks cover measurement-unit equivariance, singular-input behavior,
contamination robustness, positive definiteness, and comparison with
scikit-learn's MinCovDet when scikit-learn is installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import robustcov as rc


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def correlated_sample(seed: int, n: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(p)
    covariance = 0.6 ** np.abs(indices[:, None] - indices[None, :])
    X = rng.multivariate_normal(np.linspace(-0.5, 0.5, p), covariance, size=n)
    return X, covariance


def unit_equivariance_results() -> list[dict[str, object]]:
    X, _ = correlated_sample(seed=101, n=180, p=4)
    X[:20] += np.array([7.0, -5.0, 4.0, 6.0])
    factor = 1e-80

    cases = [
        (
            "FastMCD",
            lambda: rc.FastMCD(n_init=60, n_best=6, random_state=0),
            "covariance",
        ),
        (
            "RegularizedCauchy",
            lambda: rc.RegularizedCauchy(
                alpha=0.15,
                max_iter=300,
                tol=1e-8,
                warn_on_nonconvergence=False,
            ),
            "covariance",
        ),
        (
            "StudentTScatter",
            lambda: rc.StudentTScatter(
                df=3.0,
                alpha=0.15,
                max_iter=300,
                tol=1e-8,
                warn_on_nonconvergence=False,
            ),
            "covariance",
        ),
        (
            "MRCD",
            lambda: rc.MRCD(n_init=12, n_best=4, max_iter=30, random_state=0),
            "covariance",
        ),
        (
            "DetS",
            lambda: rc.DetS(max_iter=80),
            "covariance",
        ),
        (
            "RegularizedTyler",
            lambda: rc.RegularizedTyler(alpha=0.2, max_iter=300, tol=1e-9),
            "shape",
        ),
    ]

    results: list[dict[str, object]] = []
    for name, factory, output_kind in cases:
        base = factory().fit(X)
        scaled = factory().fit(factor * X)
        if output_kind == "shape":
            matrix_error = relative_error(scaled.shape_, base.shape_)
            location_error = relative_error(scaled.location_, factor * base.location_)
            distance_error = None
        else:
            matrix_error = relative_error(
                scaled.covariance_, factor**2 * base.covariance_
            )
            location_error = relative_error(scaled.location_, factor * base.location_)
            distance_error = relative_error(scaled.distances_, base.distances_)
        passed = (
            matrix_error < 5e-5
            and location_error < 5e-5
            and (distance_error is None or distance_error < 5e-5)
        )
        results.append(
            {
                "estimator": name,
                "output": output_kind,
                "location_relative_error": location_error,
                "matrix_relative_error": matrix_error,
                "distance_relative_error": distance_error,
                "passed": passed,
            }
        )
    return results


def contamination_curve() -> list[dict[str, object]]:
    rng = np.random.default_rng(102)
    n, p = 240, 5
    indices = np.arange(p)
    covariance = 0.6 ** np.abs(indices[:, None] - indices[None, :])
    clean = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    rows: list[dict[str, object]] = []

    for fraction in (0.0, 0.10, 0.20, 0.30):
        X = clean.copy()
        count = int(round(fraction * n))
        if count:
            X[:count] += rng.normal(loc=8.0, scale=1.0, size=(count, p))
        empirical_error = relative_error(np.cov(X, rowvar=False), covariance)
        fitted = rc.FastMCD(
            contamination=min(0.35, fraction + 0.05),
            n_init=80,
            n_best=8,
            random_state=0,
        ).fit(X)
        robust_error = relative_error(fitted.covariance_, covariance)
        rows.append(
            {
                "contamination": fraction,
                "empirical_covariance_relative_error": empirical_error,
                "fastmcd_covariance_relative_error": robust_error,
                "fastmcd_support_size": int(fitted.support_.sum()),
                "passed": bool(
                    robust_error < 0.30
                    and (fraction == 0.0 or robust_error < 0.1 * empirical_error)
                ),
            }
        )
    return rows


def reference_comparison() -> dict[str, object]:
    try:
        from sklearn.covariance import MinCovDet
    except ImportError:
        return {"available": False, "passed": True}

    rng = np.random.default_rng(103)
    p = 5
    indices = np.arange(p)
    covariance = 0.55 ** np.abs(indices[:, None] - indices[None, :])
    X = rng.multivariate_normal(np.zeros(p), covariance, size=220)
    X[:30] += np.linspace(4.0, 8.0, p)

    ours = rc.FastMCD(
        contamination=0.15, n_init=100, n_best=8, random_state=0
    ).fit(X)
    reference = MinCovDet(support_fraction=0.85, random_state=0).fit(X)
    location_error = float(
        np.linalg.norm(ours.location_ - reference.location_)
        / np.sqrt(np.trace(reference.covariance_))
    )
    covariance_error = relative_error(ours.covariance_, reference.covariance_)
    distance_correlation = float(
        np.corrcoef(ours.distances_, reference.mahalanobis(X))[0, 1]
    )
    return {
        "available": True,
        "location_relative_error": location_error,
        "covariance_relative_error": covariance_error,
        "distance_correlation": distance_correlation,
        "passed": bool(
            location_error < 0.02
            and covariance_error < 0.10
            and distance_correlation > 0.995
        ),
    }


def singular_input_results() -> dict[str, object]:
    rng = np.random.default_rng(104)
    X = np.c_[rng.normal(size=(120, 3)), np.ones(120)]
    tyler_rejected = False
    try:
        rc.TylerShape().fit(X)
    except ValueError:
        tyler_rejected = True

    regularized = rc.RegularizedTyler(alpha=0.2, max_iter=200).fit(X)
    dets = rc.DetS(max_iter=80).fit(X)
    detmm = rc.DetMM(max_iter=80).fit(X)
    eigenvalues = {
        "RegularizedTyler": float(np.linalg.eigvalsh(regularized.covariance_)[0]),
        "DetS": float(np.linalg.eigvalsh(dets.covariance_)[0]),
        "DetMM": float(np.linalg.eigvalsh(detmm.covariance_)[0]),
    }
    passed = tyler_rejected and all(value > 0.0 for value in eigenvalues.values())
    return {
        "unregularized_tyler_rejected": tyler_rejected,
        "minimum_eigenvalues": eigenvalues,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = {
        "unit_equivariance": unit_equivariance_results(),
        "contamination_curve": contamination_curve(),
        "sklearn_reference": reference_comparison(),
        "singular_inputs": singular_input_results(),
    }
    passed = (
        all(row["passed"] for row in report["unit_equivariance"])
        and all(row["passed"] for row in report["contamination_curve"])
        and report["sklearn_reference"]["passed"]
        and report["singular_inputs"]["passed"]
    )
    report["passed"] = bool(passed)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
