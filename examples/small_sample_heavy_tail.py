"""Small-sample heavy-tail estimator demo.

Run:
    python examples/small_sample_heavy_tail.py
"""
from __future__ import annotations

import time
import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n, p, df = 35, 40, 1.5
    Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    X = Z / np.sqrt(rng.chisquare(df, size=n) / df)[:, None]

    methods = {
        "RegularizedTyler": rc.RegularizedTyler(alpha=0.10, scale_correction="radial_median"),
        "KLRegularizedTyler": rc.KLRegularizedTyler(alpha=0.10, scale_correction="radial_median"),
        "StudentTScatter": rc.StudentTScatter(df=3, alpha=0.05),
        "RegularizedCauchy": rc.RegularizedCauchy(alpha=0.10),
        "HellingerRegularizedTyler(exp)": rc.HellingerRegularizedTyler(alpha=0.10, scale_correction="radial_median"),
    }

    print(f"small sample heavy-tail demo: n={n}, p={p}, df={df}")
    print("method,seconds,rel_fro_error,condition_number,radial_kurtosis")
    for name, est in methods.items():
        t0 = time.perf_counter()
        est.fit(X)
        dt = time.perf_counter() - t0
        print(f"{name},{dt:.6f},{rel_fro(est.covariance_, Sigma):.4f},{np.linalg.cond(est.covariance_):.4g},{getattr(est, 'radial_kurtosis_', float('nan')):.4f}")
