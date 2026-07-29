#!/usr/bin/env python3
"""Numerical-equivalence and acceleration gate for source separation kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import robustcov as rc


def _median_runtime(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), samples, result


def _sobi_sample(seed=0, n_samples=4000, n_features=16):
    rng = np.random.default_rng(seed)
    coefficients = np.linspace(-0.85, 0.9, n_features)
    innovations = rng.normal(size=(n_samples, n_features))
    sources = np.zeros_like(innovations)
    for index in range(1, n_samples):
        sources[index] = coefficients * sources[index - 1] + innovations[index]
    mixing = rng.normal(size=(n_features, n_features)) + n_features * np.eye(n_features)
    return sources @ mixing.T


def run_gate(repeats=5, min_speedup=1.5):
    X = _sobi_sample()
    python_time, python_samples, python_model = _median_runtime(
        lambda: rc.SOBI(lags=20, backend="python").fit(X), repeats
    )
    cpp_time, cpp_samples, cpp_model = _median_runtime(
        lambda: rc.SOBI(lags=20, backend="cpp").fit(X), repeats
    )
    speedup = python_time / cpp_time
    agreement = rc.minimum_distance_index(
        cpp_model.unmixing_, np.linalg.pinv(python_model.unmixing_)
    )
    passed = bool(speedup >= min_speedup and agreement < 1e-10)
    return {
        "case": "complete_sobi_fit",
        "python_seconds": python_time,
        "cpp_seconds": cpp_time,
        "speedup": speedup,
        "minimum_speedup": min_speedup,
        "unmixing_mdi": agreement,
        "python_samples": python_samples,
        "cpp_samples": cpp_samples,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_gate(args.repeats, args.min_speedup)
    print(json.dumps(result, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
