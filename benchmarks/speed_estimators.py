"""Speed benchmark for robustcov estimators.

Run:
    python benchmarks/speed_estimators.py --csv results/speed.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from statistics import median

import numpy as np
import robustcov as rc


def make_data(n, p, contamination, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    m = int(round(contamination * n))
    if m:
        idx = rng.choice(n, size=m, replace=False)
        X[idx] += rng.normal(8.0, 1.5, size=(m, p))
    return X


def time_fit(name, factory, X, repeat):
    times = []
    for _ in range(repeat):
        est = factory()
        t0 = time.perf_counter()
        est.fit(X)
        times.append(time.perf_counter() - t0)
    return {
        'method': name,
        'median_seconds': f"{median(times):.6f}",
        'min_seconds': f"{min(times):.6f}",
        'max_seconds': f"{max(times):.6f}",
    }


def sklearn_estimators():
    out = {}
    try:
        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        out['sklearn EmpiricalCovariance'] = lambda: EmpiricalCovariance()
        out['sklearn MinCovDet'] = lambda: MinCovDet(random_state=0, support_fraction=None)
    except Exception:
        pass
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--quality", choices=["fast", "balanced", "high"], default="fast")
    parser.add_argument("--adaptive-contamination", action="store_true",
                        help="Pass contamination to robustcov FastMCD so h≈(1-contamination)n.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path.")
    args = parser.parse_args()

    X = make_data(args.n, args.p, args.contamination, seed=0)
    adaptive = args.contamination if args.adaptive_contamination else None

    estimators = {
        'robustcov FastMCD': lambda: rc.FastMCD(quality=args.quality, n_init=args.n_init, random_state=0, contamination=adaptive),
        'robustcov TylerShape': lambda: rc.TylerShape(max_iter=200, tol=1e-6),
        'robustcov RegTyler': lambda: rc.RegularizedTyler(alpha=0.05, max_iter=200, tol=1e-6),
    }
    estimators.update(sklearn_estimators())

    print(f"n={args.n}, p={args.p}, contamination={args.contamination}, adaptive={args.adaptive_contamination}, repeat={args.repeat}")
    rows = []
    for name, factory in estimators.items():
        try:
            rows.append(time_fit(name, factory, X, args.repeat))
        except Exception as exc:
            rows.append({'method': name, 'median_seconds': 'FAILED', 'min_seconds': '', 'max_seconds': f"{type(exc).__name__}: {exc}"})

    fieldnames = ['method', 'median_seconds', 'min_seconds', 'max_seconds']
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        from pathlib import Path
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
