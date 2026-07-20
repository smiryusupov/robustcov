"""Small-sample heavy-tail covariance/scatter benchmark.

The benchmark uses the shared covariance catalog so modern robustcov estimators,
classical shrinkage baselines, and applicability constraints remain synchronized
with the speed benchmark and documentation.

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

try:
    from benchmarks.covariance_catalog import covariance_methods
except ModuleNotFoundError:  # direct script execution
    from covariance_catalog import covariance_methods


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def make_data(n, p, df, seed):
    rng = np.random.default_rng(seed)
    scatter = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    z = rng.multivariate_normal(np.zeros(p), scatter, size=n)
    if np.isinf(df):
        X = z
    else:
        # Elliptical multivariate t.  For df <= 2 the covariance is undefined,
        # so the benchmark deliberately evaluates recovery of the generating
        # scatter matrix rather than claiming covariance recovery.
        radial = rng.chisquare(df, size=n) / df
        X = z / np.sqrt(radial)[:, None]
    return X, scatter


def _summarize_numeric(values):
    vals = []
    for value in values:
        try:
            if value != "":
                vals.append(float(value))
        except Exception:
            pass
    return f"{float(np.median(vals)):.0f}" if vals else ""


def _summarize_converged(values):
    vals = [value for value in values if isinstance(value, (bool, np.bool_))]
    if not vals:
        return ""
    return f"{sum(vals)}/{len(vals)}"


PROFILES = {
    "quick": {"n": [40, 80], "p": [20, 60], "df": [1.0, 3.0]},
    "full": {"n": [30, 60, 120], "p": [20, 40, 80], "df": [1.0, 2.0, 3.0]},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--n-list", nargs="+", type=int, default=None)
    parser.add_argument("--p-list", nargs="+", type=int, default=None)
    parser.add_argument("--df-list", nargs="+", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional exact method-name filter. By default all relevant catalog methods run.",
    )
    parser.add_argument("--exclude-experimental", action="store_true")
    parser.add_argument("--exclude-selector", action="store_true")
    args = parser.parse_args()

    methods = covariance_methods(
        purpose="accuracy",
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

    profile = PROFILES[args.profile]
    n_values = args.n_list if args.n_list is not None else profile["n"]
    p_values = args.p_list if args.p_list is not None else profile["p"]
    df_values = args.df_list if args.df_list is not None else profile["df"]

    rows = []
    for n in n_values:
        for p in p_values:
            for df in df_values:
                X, scatter = make_data(n, p, df, seed=13 + n + p + int(10 * df))
                for method in methods:
                    applicable, reason = method.applicable(n, p)
                    common = {
                        "n": n,
                        "p": p,
                        "df": df,
                        "p_over_n": f"{p / n:.3f}",
                        "family": method.family,
                        "method": method.name,
                        "experimental": method.experimental,
                        "note": method.note,
                    }
                    if not applicable:
                        rows.append(
                            {
                                **common,
                                "status": "not_applicable",
                                "reason": reason,
                                "median_seconds": "",
                                "rel_fro_error": "",
                                "condition_number": "",
                                "converged": "",
                                "n_iter": "",
                                "selected_estimator": "",
                                "failures": 0,
                            }
                        )
                        continue

                    times, errors, conds = [], [], []
                    converged_values, n_iters = [], []
                    selected_names: list[str] = []
                    failures = 0
                    failure_reason = ""
                    for _ in range(args.repeat):
                        try:
                            est = method.factory()
                            t0 = time.perf_counter()
                            est.fit(X)
                            elapsed = time.perf_counter() - t0
                            cov = np.asarray(est.covariance_, dtype=float)
                            err = rel_fro(cov, scatter)
                            cond = float(np.linalg.cond(cov))
                            if not np.isfinite(err) or not np.isfinite(cond):
                                raise FloatingPointError("non-finite result")
                            times.append(elapsed)
                            errors.append(err)
                            conds.append(cond)
                            converged_values.append(getattr(est, "converged_", ""))
                            n_iters.append(getattr(est, "n_iter_", ""))
                            selected_names.append(getattr(est, "best_estimator_name_", ""))
                        except Exception as exc:
                            failures += 1
                            failure_reason = f"{type(exc).__name__}: {exc}"

                    status = "ok" if times else "failed"
                    selected = sorted({name for name in selected_names if name})
                    rows.append(
                        {
                            **common,
                            "status": status,
                            "reason": failure_reason,
                            "median_seconds": f"{float(np.median(times)):.6f}" if times else "",
                            "rel_fro_error": f"{float(np.median(errors)):.4f}" if errors else "",
                            "condition_number": f"{float(np.median(conds)):.4g}" if conds else "",
                            "converged": _summarize_converged(converged_values),
                            "n_iter": _summarize_numeric(n_iters),
                            "selected_estimator": "; ".join(selected),
                            "failures": failures,
                        }
                    )

    fieldnames = [
        "n", "p", "df", "p_over_n", "family", "method", "experimental",
        "status", "reason", "median_seconds", "rel_fro_error",
        "condition_number", "converged", "n_iter", "selected_estimator",
        "failures", "note",
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
