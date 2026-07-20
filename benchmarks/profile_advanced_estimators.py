#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Profile advanced robustcov estimators on representative complete fits.

Run with numerical-library threads pinned for comparable results::

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python benchmarks/profile_advanced_estimators.py \
      --output-dir results/advanced_estimator_profiles
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

import numpy as np

import robustcov as rc


def _datasets(seed: int = 42):
    rng = np.random.default_rng(seed)

    cellmcd = rng.standard_normal((360, 12))
    cellmcd[:25, :4] += 6.0
    cellmcd[rng.random(cellmcd.shape) < 0.02] = np.nan

    cellpca = rng.standard_normal((900, 45))
    cellpca[:45, :8] += 5.0
    cellpca[rng.random(cellpca.shape) < 0.015] = np.nan

    tensor = rng.standard_normal((260, 12, 10))
    tensor[:25, :4, :4] += 4.0
    tensor[rng.random(tensor.shape) < 0.01] = np.nan

    graph = rng.standard_normal((900, 42))
    graph[:45, :8] += 4.0

    kernel = rng.standard_normal((420, 16))
    kernel[:35, :5] += 4.0
    return cellmcd, cellpca, tensor, graph, kernel


def _cases(seed: int = 42):
    cellmcd, cellpca, tensor, graph, kernel = _datasets(seed)
    return {
        "cellmcd": (
            rc.CellMCD(
                alpha=0.75,
                max_iter=12,
                tol=1e-5,
                min_samples_per_feature=None,
            ),
            cellmcd,
        ),
        "cellpca": (
            rc.CellwiseRobustPCA(
                n_components=6,
                max_iter=20,
                tol=1e-6,
            ),
            cellpca,
        ),
        "multilinear_pca": (
            rc.RobustMultilinearPCA(
                ranks=(3, 3),
                max_iter=15,
                tol=1e-6,
                backend="auto",
            ),
            tensor,
        ),
        "sparse_precision": (
            rc.RobustGraphicalLasso(
                alpha=0.08,
                scatter_estimator="empirical",
                standardize=True,
                max_iter=180,
            ),
            graph,
        ),
        "kernel_mrcd": (
            rc.KernelMRCD(
                n_init=12,
                n_best=4,
                initial_c_steps=2,
                max_iter=20,
                random_state=7,
            ),
            kernel,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = _cases(args.seed)
    if args.case != "all":
        if args.case not in cases:
            parser.error("--case must be 'all' or one of: " + ", ".join(cases))
        cases = {args.case: cases[args.case]}
    if args.top < 1:
        parser.error("--top must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (estimator, X) in cases.items():
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        estimator.fit(X)
        profiler.disable()
        elapsed = time.perf_counter() - started

        profile_path = args.output_dir / f"{name}.prof"
        text_path = args.output_dir / f"{name}.txt"
        profiler.dump_stats(profile_path)
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(args.top)
            stream.write("\n--- internal time ---\n")
            stats.sort_stats("tottime").print_stats(args.top)

        summary[name] = {
            "elapsed_seconds": elapsed,
            "n_samples": int(X.shape[0]),
            "shape": list(X.shape[1:]),
            "n_iter": getattr(estimator, "n_iter_", None),
            "profile": str(profile_path),
            "text": str(text_path),
        }
        print(f"{name}: {elapsed:.6f}s -> {text_path}")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
