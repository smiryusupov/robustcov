"""FastMCD speed/accuracy tradeoff benchmark.

Run:
    python benchmarks/fastmcd_quality_speed_tradeoff.py --n 1000 --p 8 --contamination 0.2
    python benchmarks/fastmcd_quality_speed_tradeoff.py --n 1000 --p 8 --contamination 0.3 --adaptive-contamination
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def make_data(n, p, contamination, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    m = int(round(contamination * n))
    if m:
        idx = rng.choice(n, size=m, replace=False)
        X[idx] += rng.normal(8.0, 1.5, size=(m, p))
    return X, Sigma


def time_and_error(name, factory, X, Sigma, repeat):
    times = []
    err = None
    h = ""
    support = ""
    raw_scale = ""
    for _ in range(repeat):
        est = factory()
        t0 = time.perf_counter()
        est.fit(X)
        times.append(time.perf_counter() - t0)
        err = rel_fro(est.covariance_, Sigma)
        h = getattr(est, "h_", "")
        supp = getattr(est, "support_", None)
        support = int(supp.sum()) if supp is not None else ""
        raw_scale = getattr(est, "raw_scale_", "")
    return {
        "method": name,
        "median_seconds": f"{float(np.median(times)):.6f}",
        "rel_fro_error": f"{err:.4f}",
        "h_raw": h,
        "support_size": support,
        "raw_scale": f"{raw_scale:.4f}" if isinstance(raw_scale, float) else raw_scale,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--contamination", type=float, default=0.20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--adaptive-contamination", action="store_true",
                        help="Pass contamination to robustcov FastMCD so h≈(1-contamination)n.")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    X, Sigma = make_data(args.n, args.p, args.contamination, seed=42)
    adaptive = args.contamination if args.adaptive_contamination else None

    methods = {
        "robustcov FastMCD fast": lambda: rc.FastMCD(quality="fast", random_state=0, scale_correction="none", contamination=adaptive),
        "robustcov FastMCD balanced": lambda: rc.FastMCD(quality="balanced", random_state=0, scale_correction="none", contamination=adaptive),
        "robustcov FastMCD high": lambda: rc.FastMCD(quality="high", random_state=0, scale_correction="none", contamination=adaptive),
    }
    try:
        from sklearn.covariance import MinCovDet
        methods["sklearn MinCovDet"] = lambda: MinCovDet(random_state=0, support_fraction=None)
    except Exception:
        pass

    print(f"n={args.n}, p={args.p}, contamination={args.contamination}, adaptive={args.adaptive_contamination}, repeat={args.repeat}")
    rows = []
    for name, factory in methods.items():
        try:
            rows.append(time_and_error(name, factory, X, Sigma, args.repeat))
        except Exception as exc:
            rows.append({"method": name, "median_seconds": "FAILED", "rel_fro_error": f"{type(exc).__name__}: {exc}", "h_raw": "", "support_size": "", "raw_scale": ""})

    fieldnames = ["method", "median_seconds", "rel_fro_error", "h_raw", "support_size", "raw_scale"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
