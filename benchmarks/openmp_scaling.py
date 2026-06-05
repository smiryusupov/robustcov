#!/usr/bin/env python
"""OpenMP scaling benchmark for C++ robustcov kernels.

This benchmark is intentionally simple: it runs FastMCD and RegularizedTyler with
several thread counts and reports median runtime. Speedups depend on compiler,
BLAS settings, CPU, n, p, and estimator settings.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import robustcov as rc


def timed(fn, repeat: int):
    vals = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    vals = np.asarray(vals)
    return float(np.median(vals)), float(vals.min()), float(vals.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(args.n, args.p))
    X[: max(1, args.n // 10)] += 8.0

    print(f"openmp_enabled={rc.has_openmp()}, default_threads={rc.get_num_threads()}")
    print("method,threads,median_seconds,min_seconds,max_seconds,speedup_vs_1")

    rows = []
    baselines: dict[str, float] = {}
    for method in ["FastMCD", "RegularizedTyler"]:
        for th in args.threads:
            if method == "FastMCD":
                fn = lambda th=th: rc.FastMCD(n_init=250, n_best=8, quality="fast", random_state=0, n_jobs=th).fit(X)
            else:
                fn = lambda th=th: rc.RegularizedTyler(alpha=0.1, max_iter=80, n_jobs=th).fit(X)
            med, mn, mx = timed(fn, args.repeat)
            if th == 1 or method not in baselines:
                baselines.setdefault(method, med)
            speedup = baselines[method] / med if med > 0 else float("nan")
            row = {
                "method": method,
                "threads": th,
                "median_seconds": med,
                "min_seconds": mn,
                "max_seconds": mx,
                "speedup_vs_1": speedup,
            }
            rows.append(row)
            print(f"{method},{th},{med:.6f},{mn:.6f},{mx:.6f},{speedup:.3f}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
