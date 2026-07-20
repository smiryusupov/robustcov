#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Profile cellwise covariance, sparse PCA, selection, and monitoring workflows.

Run with numerical-library threads pinned for comparable results::

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python benchmarks/profile_workflow_estimators.py \
      --output-dir results/workflow_profiles
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

    cellrcov = rng.standard_normal((650, 55))
    cellrcov[:40, :8] += 5.0
    cellrcov[rng.random(cellrcov.shape) < 0.015] = np.nan

    sparse = rng.standard_normal((600, 160))
    sparse[:40, :20] += 4.0
    sparse[rng.random(sparse.shape) < 0.015] = np.nan

    automatic = rng.standard_t(df=3, size=(700, 24))
    automatic[:50, :6] += 4.0

    reference = rng.standard_normal((1200, 26))
    reference[:50, :5] += 2.5

    stream = rng.standard_normal((600, 26))
    stream[300:, :4] += 1.5
    return cellrcov, sparse, automatic, reference, stream


def _cases(seed: int = 42):
    cellrcov, sparse, automatic, reference, stream = _datasets(seed)

    cell_pca = rc.CellwiseRobustPCA(
        n_components=6,
        max_iter=15,
        tol=1e-6,
    )

    def monitor_updates():
        monitor = rc.RobustSubspaceMonitor(
            n_components=6,
            window_size=180,
            calibration_windows=5,
            random_state=7,
        ).fit(reference)
        for batch in np.array_split(stream, 12):
            monitor.update(batch)
        return monitor

    return {
        "cellrcov": lambda: rc.CellRCov(
            n_components=6,
            cell_pca=cell_pca,
            cv_splits=5,
            store_diagnostics=True,
        ).fit(cellrcov),
        "sparse_cellpca": lambda: rc.SparseCellPCA(
            n_components=8,
            max_iter=10,
            tol=1e-6,
            alpha=0.04,
            loading_max_iter=80,
            loading_tol=1e-7,
        ).fit(sparse),
        "auto_scatter": lambda: rc.AutoRobustScatter(
            selection="stability",
            n_splits=2,
            subsample_fraction=0.7,
            random_state=7,
        ).fit(automatic),
        "monitor_fit": lambda: rc.RobustSubspaceMonitor(
            n_components=6,
            window_size=180,
            calibration_windows=8,
            random_state=7,
        ).fit(reference),
        "monitor_updates": monitor_updates,
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
    for name, run in cases.items():
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        estimator = run()
        profiler.disable()
        elapsed = time.perf_counter() - started

        profile_path = args.output_dir / f"{name}.prof"
        text_path = args.output_dir / f"{name}.txt"
        profiler.dump_stats(profile_path)
        with text_path.open("w", encoding="utf-8") as stream_handle:
            stats = pstats.Stats(profiler, stream=stream_handle).strip_dirs()
            stats.sort_stats("cumulative").print_stats(args.top)
            stream_handle.write("\n--- internal time ---\n")
            stats.sort_stats("tottime").print_stats(args.top)

        summary[name] = {
            "elapsed_seconds": elapsed,
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
