#!/usr/bin/env python3
"""Profile representative robustcov estimator fits.

Run with BLAS/OpenMP threads pinned for reproducible comparisons, e.g.::

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python benchmarks/profile_estimators.py --output-dir results/round3_profiles
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np

import robustcov as rc
from robustcov.m_estimators import RegularizedCauchy
from robustcov.pca import RobustPCA


def _datasets(seed: int = 42):
    rng = np.random.default_rng(seed)

    vector = rng.standard_normal((4_000, 24))
    vector[:200, :8] += 4.0

    high_dimensional = rng.standard_normal((180, 120))
    high_dimensional[:18, :20] += 5.0

    matrix = rng.standard_normal((220, 10, 10))
    matrix[:20, :4, :4] += 4.0

    pca = rng.standard_normal((4_000, 64))
    pca[:200, :10] += 4.0
    return vector, high_dimensional, matrix, pca


def _cases():
    vector, high_dimensional, matrix, pca = _datasets()
    return {
        "fastmcd": (
            rc.FastMCD(
                quality="fast",
                n_init=100,
                n_best=5,
                max_iter=50,
                random_state=7,
                n_jobs=1,
            ),
            vector,
        ),
        "tyler": (
            rc.TylerShape(max_iter=300, tol=1e-8, n_jobs=1),
            vector,
        ),
        "mrcd": (
            rc.MRCD(
                quality="fast",
                n_init=20,
                n_best=5,
                initial_c_steps=2,
                max_iter=30,
                standardization="qn",
                random_state=7,
            ),
            high_dimensional,
        ),
        "matrix_mcd": (
            rc.MatrixMCD(
                n_init=20,
                n_best=4,
                initial_c_steps=2,
                max_iter=20,
                flip_flop_initial_iter=2,
                flip_flop_max_iter=30,
                random_state=7,
                backend="auto",
            ),
            matrix,
        ),
        "robust_pca": (
            RobustPCA(
                n_components=12,
                estimator=RegularizedCauchy(
                    alpha=0.1, max_iter=150, tol=1e-7
                ),
                store_scores=True,
            ),
            pca,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=["all", *_cases().keys()], default="all")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = _cases()
    if args.case != "all":
        selected = {args.case: selected[args.case]}

    summary = {}
    for name, (estimator, X) in selected.items():
        profile_path = args.output_dir / f"{name}.prof"
        text_path = args.output_dir / f"{name}.txt"
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        estimator.fit(X)
        profiler.disable()
        elapsed = time.perf_counter() - started
        profiler.dump_stats(profile_path)
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumulative").print_stats(args.top)
            stream.write("\n--- internal time ---\n")
            stats.sort_stats("tottime").print_stats(args.top)
        summary[name] = {
            "elapsed_seconds": elapsed,
            "profile": str(profile_path),
            "text": str(text_path),
            "n_samples": int(X.shape[0]),
            "shape": list(X.shape[1:]),
        }
        print(f"{name}: {elapsed:.6f}s -> {text_path}")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
