"""Accuracy benchmark: covariance error as contamination increases.

Run:
    python benchmarks/accuracy_vs_contamination.py
    python benchmarks/accuracy_vs_contamination.py --adaptive-contamination
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


def maybe_sklearn():
    try:
        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        return {
            "sklearn Empirical": lambda X, frac: EmpiricalCovariance().fit(X),
            "sklearn MinCovDet": lambda X, frac: MinCovDet(random_state=0).fit(X),
        }
    except Exception:
        class Emp:
            def fit(self, X):
                self.covariance_ = np.cov(X, rowvar=False)
                return self
        return {"Empirical": lambda X, frac: Emp().fit(X)}


def make_clean(n, p, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    return X, Sigma, rng


def contaminate(X, frac, rng):
    X = X.copy()
    m = int(round(frac * X.shape[0]))
    if m:
        idx = rng.choice(X.shape[0], size=m, replace=False)
        X[idx] += rng.normal(8.0, 1.5, size=(m, X.shape[1]))
    return X


def summarize_estimator(est):
    h = getattr(est, "h_", "")
    support = getattr(est, "support_", None)
    support_size = int(support.sum()) if support is not None else ""
    return h, support_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--quality", choices=["fast", "balanced", "high"], default="balanced")
    parser.add_argument("--adaptive-contamination", action="store_true",
                        help="Pass the known contamination fraction to robustcov FastMCD so h≈(1-contamination)n.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path.")
    args = parser.parse_args()

    X_clean, Sigma, rng = make_clean(args.n, args.p, seed=42)

    def fit_fastmcd(X, frac):
        contamination = frac if args.adaptive_contamination else None
        return rc.FastMCD(
            quality=args.quality,
            n_init=args.n_init,
            random_state=0,
            scale_correction="none",
            contamination=contamination,
        ).fit(X)

    methods = {
        "robustcov FastMCD": fit_fastmcd,
        "robustcov RegTyler": lambda X, frac: rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median").fit(X),
    }
    methods.update(maybe_sklearn())

    rows = []
    for frac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
        X = contaminate(X_clean, frac, rng)
        for name, fit_est in methods.items():
            t0 = time.perf_counter()
            try:
                est = fit_est(X, frac)
                dt = time.perf_counter() - t0
                h, support_size = summarize_estimator(est)
                rows.append({
                    "contamination": f"{frac:.2f}",
                    "method": name,
                    "rel_fro_error": f"{rel_fro(est.covariance_, Sigma):.4f}",
                    "seconds": f"{dt:.6f}",
                    "h_raw": h,
                    "support_size": support_size,
                    "adaptive": args.adaptive_contamination,
                })
            except Exception as exc:
                rows.append({
                    "contamination": f"{frac:.2f}",
                    "method": name,
                    "rel_fro_error": "FAILED",
                    "seconds": f"{type(exc).__name__}: {exc}",
                    "h_raw": "",
                    "support_size": "",
                    "adaptive": args.adaptive_contamination,
                })

    fieldnames = ["contamination", "method", "rel_fro_error", "seconds", "h_raw", "support_size", "adaptive"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
