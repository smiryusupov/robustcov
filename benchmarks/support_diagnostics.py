"""Synthetic support diagnostics for FastMCD.

Run:
    python benchmarks/support_diagnostics.py --n 1000 --p 8 --contamination 0.3 --csv results/support.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def make_data(n, p, contamination, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X_clean = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    X = X_clean.copy()
    m = int(round(contamination * n))
    is_outlier = np.zeros(n, dtype=bool)
    if m:
        idx = rng.choice(n, size=m, replace=False)
        is_outlier[idx] = True
        X[idx] += rng.normal(8.0, 1.5, size=(m, p))
    return X, X_clean, Sigma, is_outlier


def support_metrics(support, is_outlier):
    support = np.asarray(support, dtype=bool)
    clean = ~is_outlier
    kept_clean = np.sum(support & clean)
    kept_out = np.sum(support & is_outlier)
    support_size = np.sum(support)
    clean_total = np.sum(clean)
    purity = kept_clean / max(1, support_size)
    clean_recall = kept_clean / max(1, clean_total)
    outlier_leakage = kept_out / max(1, np.sum(is_outlier))
    return purity, clean_recall, outlier_leakage, kept_clean, kept_out, support_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--contamination", type=float, default=0.30)
    parser.add_argument("--quality", choices=["fast", "balanced", "high"], default="balanced")
    parser.add_argument("--adaptive-contamination", action="store_true")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    X, X_clean, Sigma, is_outlier = make_data(args.n, args.p, args.contamination, seed=42)
    contamination_arg = args.contamination if args.adaptive_contamination else None

    methods = {
        f"robustcov FastMCD {args.quality}": lambda: rc.FastMCD(
            quality=args.quality,
            random_state=0,
            contamination=contamination_arg,
            scale_correction="none",
        ),
    }
    try:
        from sklearn.covariance import MinCovDet
        methods["sklearn MinCovDet"] = lambda: MinCovDet(random_state=0, support_fraction=None)
    except Exception:
        pass

    rows = []
    for name, factory in methods.items():
        est = factory()
        t0 = time.perf_counter()
        est.fit(X)
        seconds = time.perf_counter() - t0
        support = getattr(est, "support_", np.zeros(args.n, dtype=bool))
        purity, clean_recall, leakage, kept_clean, kept_out, support_size = support_metrics(support, is_outlier)
        support_cov_error = ""
        if support_size > args.p + 1:
            support_cov = np.cov(X[np.asarray(support, dtype=bool)], rowvar=False)
            support_cov_error = f"{rel_fro(support_cov, Sigma):.4f}"
        rows.append({
            "method": name,
            "seconds": f"{seconds:.6f}",
            "rel_fro_error": f"{rel_fro(est.covariance_, Sigma):.4f}",
            "support_cov_error": support_cov_error,
            "h_raw": getattr(est, "h_", ""),
            "support_size": int(support_size),
            "support_purity": f"{purity:.4f}",
            "clean_recall": f"{clean_recall:.4f}",
            "outlier_leakage": f"{leakage:.4f}",
            "kept_clean": int(kept_clean),
            "kept_outliers": int(kept_out),
        })

    fieldnames = [
        "method", "seconds", "rel_fro_error", "support_cov_error", "h_raw", "support_size",
        "support_purity", "clean_recall", "outlier_leakage", "kept_clean", "kept_outliers",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
