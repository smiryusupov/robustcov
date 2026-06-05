"""Benchmark AutoRobustScatter on small-sample heavy-tailed data.

Run:
    python benchmarks/auto_scatter_small_sample.py --selection stability --csv results/auto_scatter.csv
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


def make_data(n, p, df, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    g = rng.chisquare(df, size=n) / df
    X = Z / np.sqrt(g)[:, None]
    return X, Sigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument("--p-list", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--df-list", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--selection", choices=["diagnostic", "stability"], default="stability")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    rows = []
    for n in args.n_list:
        for p in args.p_list:
            for df in args.df_list:
                X, Sigma = make_data(n, p, df, seed=222 + n + p + int(10 * df))
                t0 = time.perf_counter()
                est = rc.AutoRobustScatter(selection=args.selection, n_splits=args.n_splits).fit(X)
                seconds = time.perf_counter() - t0
                br = est.best_result_
                rows.append({
                    "n": n,
                    "p": p,
                    "df": df,
                    "p_over_n": f"{p/n:.3f}",
                    "selection": args.selection,
                    "selected": est.best_estimator_name_,
                    "score": f"{br.score:.4f}",
                    "diagnostic_score": f"{br.diagnostic_score:.4f}",
                    "stability_score": f"{br.stability_score:.4f}",
                    "seconds": f"{seconds:.6f}",
                    "rel_fro_error": f"{rel_fro(est.covariance_, Sigma):.4f}",
                    "condition_number": f"{np.linalg.cond(est.covariance_):.4g}",
                    "radial_kurtosis": f"{getattr(est, 'radial_kurtosis_', float('nan')):.4g}",
                    "converged": getattr(est, "converged_", ""),
                    "n_iter": getattr(est, "n_iter_", ""),
                })

    fieldnames = list(rows[0])
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
