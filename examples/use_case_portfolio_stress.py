"""Use case: portfolio covariance stress comparison.

This example compares empirical covariance and RegularizedCauchy under simulated
market stress days. It reports the largest eigenvalue and a simple equal-weight
portfolio risk estimate.

Run:
    python examples/use_case_portfolio_stress.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def portfolio_risk(cov):
    p = cov.shape[0]
    w = np.ones(p) / p
    return float(np.sqrt(w @ cov @ w))


if __name__ == "__main__":
    rng = np.random.default_rng(26)
    n, p, df = 120, 40, 2.5
    Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n) * np.sqrt(df / rng.chisquare(df, size=n))[:, None]
    stress = rng.choice(n, size=10, replace=False)
    market = rng.normal(0, 5.0, size=(stress.size, 1))
    X[stress] += market * np.linspace(0.5, 1.5, p)[None, :]

    emp = np.cov(X, rowvar=False)
    cauchy = rc.RegularizedCauchy(alpha=0.10, warn_on_nonconvergence=False).fit(X)

    print("portfolio covariance stress comparison")
    print(f"empirical_risk={portfolio_risk(emp):.4f}, empirical_cond={np.linalg.cond(emp):.4g}")
    print(f"cauchy_risk={portfolio_risk(cauchy.covariance_):.4f}, cauchy_cond={np.linalg.cond(cauchy.covariance_):.4g}")
    print(f"cauchy_radial_kurtosis={cauchy.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/portfolio_stress")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_covariance_heatmap(cauchy.covariance_, title="RegularizedCauchy portfolio covariance", output_path=outdir / "covariance.png", show=False)
    rc.plot_robust_distance_panel(cauchy, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
