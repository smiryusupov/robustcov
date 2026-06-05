"""Use case: heavy-tailed finance-style covariance estimation.

The example simulates heavy-tailed asset returns in a small-sample regime. The goal is
not to forecast markets; it shows why robust shrinkage scatter can be preferable to
empirical covariance when returns are heavy-tailed and n is not much larger than p.

Run:
    python examples/use_case_finance_risk.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n, p, df = 80, 50, 2.0
    Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    scale = np.sqrt(df / rng.chisquare(df, size=n))[:, None]
    X = Z * scale

    estimators = {
        "Empirical": lambda: type("Emp", (), {"fit": lambda self, X: setattr(self, "covariance_", np.cov(X, rowvar=False)) or self})(),
        "RegularizedCauchy": lambda: rc.RegularizedCauchy(alpha=0.10, warn_on_nonconvergence=False),
        "StudentTScatter": lambda: rc.StudentTScatter(df=3, alpha=0.05, warn_on_nonconvergence=False),
        "RegularizedTyler": lambda: rc.RegularizedTyler(alpha=0.10, scale_correction="radial_median"),
    }

    print("finance-style heavy-tail covariance demo")
    print(f"n={n}, p={p}, df={df}")
    print("method, rel_fro_error, condition_number, radial_kurtosis")
    best = None
    for name, factory in estimators.items():
        est = factory().fit(X)
        cov = est.covariance_
        rk = getattr(est, "radial_kurtosis_", np.nan)
        cond = np.linalg.cond(cov)
        err = rel_fro(cov, Sigma)
        print(f"{name}, {err:.4f}, {cond:.4g}, {rk:.4g}")
        if name == "RegularizedCauchy":
            best = est

    outdir = Path("results/use_cases/finance")
    outdir.mkdir(parents=True, exist_ok=True)
    if best is not None:
        rc.plot_robust_distance_panel(best, output_path=outdir / "distance_panel.png", show=False)
        rc.plot_covariance_heatmap(best.covariance_, title="RegularizedCauchy covariance", output_path=outdir / "covariance.png", show=False)
        print("saved diagnostics to", outdir)
