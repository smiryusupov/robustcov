"""Small-sample heavy-tail benchmark.

This benchmark stresses regimes where p is close to n or p > n. It compares
classical/shrinkage covariance with robust regularized scatter estimators.

Run:
    python benchmarks/small_sample_heavy_tail.py --csv results/small_sample.csv
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
    if np.isinf(df):
        X = Z
    else:
        # multivariate t via Gaussian divided by sqrt(chi2/df)
        g = rng.chisquare(df, size=n) / df
        X = Z / np.sqrt(g)[:, None]
    return X, Sigma


def sklearn_methods():
    out = {}
    try:
        from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, MinCovDet
        out["sklearn Empirical"] = lambda: EmpiricalCovariance()
        out["sklearn LedoitWolf"] = lambda: LedoitWolf()
        out["sklearn OAS"] = lambda: OAS()
        out["sklearn MinCovDet"] = lambda: MinCovDet(random_state=0)
    except Exception:
        pass
    return out


def _summarize_numeric(values):
    vals = []
    for v in values:
        try:
            if v != "":
                vals.append(float(v))
        except Exception:
            pass
    return f"{float(np.median(vals)):.0f}" if vals else ""


def _summarize_converged(values):
    vals = [v for v in values if isinstance(v, (bool, np.bool_))]
    if not vals:
        return ""
    return f"{sum(vals)}/{len(vals)}"


def robustcov_methods():
    return {
        "robustcov RegTyler": lambda: rc.RegularizedTyler(alpha=0.10, scale_correction="radial_median", max_iter=300),
        "robustcov KLTyler": lambda: rc.KLRegularizedTyler(alpha=0.10, scale_correction="radial_median", max_iter=300),
        "robustcov StudentT(df=3)": lambda: rc.StudentTScatter(df=3, alpha=0.05, max_iter=500, damping=0.7, tol=1e-5, warn_on_nonconvergence=False),
        "robustcov Cauchy": lambda: rc.RegularizedCauchy(alpha=0.10, max_iter=500, damping=0.7, tol=1e-5, warn_on_nonconvergence=False),
        "robustcov HellTyler(exp)": lambda: rc.HellingerRegularizedTyler(alpha=0.10, scale_correction="radial_median", max_iter=150),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument("--p-list", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--df-list", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    rows = []
    for n in args.n_list:
        for p in args.p_list:
            for df in args.df_list:
                X, Sigma = make_data(n, p, df, seed=13 + n + p + int(10 * df))
                methods = {}
                methods.update(robustcov_methods())
                methods.update(sklearn_methods())
                for name, factory in methods.items():
                    times, errors, conds, converged_values, n_iters, failures = [], [], [], [], [], 0
                    for _ in range(args.repeat):
                        try:
                            est = factory()
                            t0 = time.perf_counter()
                            est.fit(X)
                            dt = time.perf_counter() - t0
                            cov = np.asarray(est.covariance_, dtype=float)
                            err = rel_fro(cov, Sigma)
                            cond = float(np.linalg.cond(cov))
                            if not np.isfinite(err) or not np.isfinite(cond):
                                raise FloatingPointError("non-finite result")
                            times.append(dt)
                            errors.append(err)
                            conds.append(cond)
                            conv = getattr(est, "converged_", "")
                            converged_values.append(conv)
                            n_iters.append(getattr(est, "n_iter_", ""))
                        except Exception:
                            failures += 1
                    rows.append({
                        "n": n,
                        "p": p,
                        "df": df,
                        "p_over_n": f"{p / n:.3f}",
                        "method": name,
                        "median_seconds": f"{float(np.median(times)):.6f}" if times else "FAILED",
                        "rel_fro_error": f"{float(np.median(errors)):.4f}" if errors else "FAILED",
                        "condition_number": f"{float(np.median(conds)):.4g}" if conds else "FAILED",
                        "converged": _summarize_converged(converged_values),
                        "n_iter": _summarize_numeric(n_iters),
                        "failures": failures,
                    })

    fieldnames = ["n", "p", "df", "p_over_n", "method", "median_seconds", "rel_fro_error", "condition_number", "converged", "n_iter", "failures"]
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
