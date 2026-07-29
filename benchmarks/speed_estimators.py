"""Workload-aware speed benchmark for covariance and scatter estimators.

Unlike the original snapshot, this benchmark uses the shared covariance catalog
and reports separate workloads instead of implying that one low-dimensional MCD
case represents every estimator family.

Run:
    python benchmarks/speed_estimators.py --profile quick --csv results/speed.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

try:
    from benchmarks.covariance_catalog import covariance_methods
except ModuleNotFoundError:  # direct script execution
    from covariance_catalog import covariance_methods


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    n: int
    p: int
    contamination: float
    df: float
    seed: int


PROFILES = {
    "quick": [
        Workload("classical_contamination", "Low-dimensional row contamination", 800, 8, 0.10, np.inf, 10),
        Workload("heavy_tail", "Moderate-dimensional Student-t tails", 500, 20, 0.00, 3.0, 20),
        Workload("high_dimensional", "High-dimensional heavy tails (p > n)", 90, 120, 0.00, 3.0, 30),
    ],
    "full": [
        Workload("classical_contamination", "Low-dimensional row contamination", 3000, 15, 0.10, np.inf, 10),
        Workload("heavy_tail", "Moderate-dimensional Student-t tails", 1500, 40, 0.00, 3.0, 20),
        Workload("high_dimensional", "High-dimensional heavy tails (p > n)", 250, 350, 0.00, 3.0, 30),
    ],
}


def make_data(workload: Workload) -> np.ndarray:
    rng = np.random.default_rng(workload.seed)
    scatter = 0.5 ** np.abs(np.subtract.outer(np.arange(workload.p), np.arange(workload.p)))
    X = rng.multivariate_normal(np.zeros(workload.p), scatter, size=workload.n)
    if np.isfinite(workload.df):
        radial = rng.chisquare(workload.df, size=workload.n) / workload.df
        X = X / np.sqrt(radial)[:, None]
    m = int(round(workload.contamination * workload.n))
    if m:
        idx = rng.choice(workload.n, size=m, replace=False)
        X[idx] += rng.normal(8.0, 1.5, size=(m, workload.p))
    return X


def time_fit(factory, X: np.ndarray, repeat: int):
    times = []
    selected = []
    for _ in range(repeat):
        estimator = factory()
        start = time.perf_counter()
        estimator.fit(X)
        times.append(time.perf_counter() - start)
        selected.append(getattr(estimator, "best_estimator_name_", ""))
    return times, sorted({name for name in selected if name})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=[workload.key for workload in PROFILES["quick"]],
        default=None,
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path.")
    parser.add_argument("--exclude-experimental", action="store_true")
    parser.add_argument("--exclude-selector", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional exact method-name filter.",
    )
    # Backward-compatible single-workload overrides used by older commands.
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--quality", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-init", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--adaptive-contamination", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be positive")

    workloads = list(PROFILES[args.profile])
    if args.workloads:
        selected_keys = set(args.workloads)
        workloads = [workload for workload in workloads if workload.key in selected_keys]
    if args.n is not None or args.p is not None or args.contamination is not None:
        base = workloads[0]
        workloads = [
            Workload(
                "custom",
                "Custom covariance workload",
                args.n if args.n is not None else base.n,
                args.p if args.p is not None else base.p,
                args.contamination if args.contamination is not None else base.contamination,
                np.inf,
                base.seed,
            )
        ]

    methods = covariance_methods(
        purpose="speed",
        include_experimental=not args.exclude_experimental,
        include_selector=not args.exclude_selector,
        include_sklearn=True,
    )
    if args.methods:
        requested = set(args.methods)
        methods = [method for method in methods if method.name in requested]
        missing = sorted(requested - {method.name for method in methods})
        if missing:
            parser.error(f"unknown or unavailable methods: {', '.join(missing)}")

    rows = []
    for workload in workloads:
        X = make_data(workload)
        print(
            f"workload={workload.label!r}, n={workload.n}, p={workload.p}, "
            f"contamination={workload.contamination}, df={workload.df}, repeat={args.repeat}"
        )
        for method in methods:
            applicable, reason = method.applicable(workload.n, workload.p)
            row = {
                "workload": workload.key,
                "workload_label": workload.label,
                "n": workload.n,
                "p": workload.p,
                "contamination": workload.contamination,
                "df": workload.df,
                "family": method.family,
                "method": method.name,
                "experimental": method.experimental,
                "note": method.note,
            }
            if not applicable:
                rows.append(
                    {
                        **row,
                        "status": "not_applicable",
                        "reason": reason,
                        "median_seconds": "",
                        "min_seconds": "",
                        "max_seconds": "",
                        "selected_estimator": "",
                    }
                )
                continue
            try:
                times, selected = time_fit(method.factory, X, args.repeat)
                rows.append(
                    {
                        **row,
                        "status": "ok",
                        "reason": "",
                        "median_seconds": f"{median(times):.6f}",
                        "min_seconds": f"{min(times):.6f}",
                        "max_seconds": f"{max(times):.6f}",
                        "selected_estimator": "; ".join(selected),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **row,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "median_seconds": "",
                        "min_seconds": "",
                        "max_seconds": "",
                        "selected_estimator": "",
                    }
                )

    fieldnames = [
        "workload", "workload_label", "n", "p", "contamination", "df",
        "family", "method", "experimental", "status", "reason",
        "median_seconds", "min_seconds", "max_seconds", "selected_estimator",
        "note",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            file_writer.writeheader()
            file_writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
