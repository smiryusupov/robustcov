"""Grid benchmark against sklearn over n, p, and contamination.

Run:
    python benchmarks/sklearn_comparison_grid.py --repeat 2 --csv benchmark_results.csv
    python benchmarks/sklearn_comparison_grid.py --repeat 2 --adaptive-contamination
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


def fit_once(factory, X):
    t0 = time.perf_counter()
    est = factory().fit(X)
    return est, time.perf_counter() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--adaptive-contamination", action="store_true")
    args = parser.parse_args()

    def methods_for(contamination):
        adaptive = contamination if args.adaptive_contamination else None
        methods = {
            "robustcov FastMCD fast": lambda: rc.FastMCD(quality="fast", random_state=0, contamination=adaptive),
            "robustcov FastMCD balanced": lambda: rc.FastMCD(quality="balanced", random_state=0, contamination=adaptive),
            "robustcov RegTyler": lambda: rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median"),
        }
        try:
            from sklearn.covariance import EmpiricalCovariance, MinCovDet
            methods["sklearn Empirical"] = lambda: EmpiricalCovariance()
            methods["sklearn MinCovDet"] = lambda: MinCovDet(random_state=0)
        except Exception:
            pass
        return methods

    rows = []
    for n in [500, 1000, 2000]:
        for p in [5, 10, 20]:
            if n <= 3 * p:
                continue
            for contamination in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
                X, Sigma = make_data(n, p, contamination, seed=100 + n + p + int(100 * contamination))
                for name, factory in methods_for(contamination).items():
                    times = []
                    errors = []
                    supports = []
                    hs = []
                    for _ in range(args.repeat):
                        try:
                            est, dt = fit_once(factory, X)
                            times.append(dt)
                            errors.append(rel_fro(est.covariance_, Sigma))
                            supp = getattr(est, "support_", None)
                            supports.append(int(supp.sum()) if supp is not None else np.nan)
                            hs.append(getattr(est, "h_", np.nan))
                        except Exception:
                            times.append(float("nan"))
                            errors.append(float("nan"))
                            supports.append(float("nan"))
                            hs.append(float("nan"))
                    rows.append({
                        "n": n,
                        "p": p,
                        "contamination": contamination,
                        "method": name,
                        "median_seconds": float(np.nanmedian(times)),
                        "rel_fro_error": float(np.nanmedian(errors)),
                        "h_raw": float(np.nanmedian(hs)),
                        "support_size": float(np.nanmedian(supports)),
                        "adaptive": args.adaptive_contamination,
                    })

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
