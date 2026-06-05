"""Use case: quality-control monitoring with robust covariance.

A manufacturing or lab process may have correlated measurement channels. Robust covariance
lets you estimate the normal operating region even when some batches are abnormal.

Run:
    python examples/use_case_quality_control.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n, p = 500, 4
    Sigma = np.array([
        [1.0, 0.6, 0.2, 0.1],
        [0.6, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0],
    ])
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    y = np.zeros(n, dtype=int)
    bad = rng.choice(n, size=45, replace=False)
    y[bad] = 1
    X[bad] += np.array([3.0, 3.0, -2.0, 1.5])

    est = rc.FastMCD(quality="balanced", random_state=0).fit(X)
    report = rc.diagnostic_report(est)
    print(report.summary())

    outdir = Path("results/use_cases/quality_control")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_anomaly_scatter_2d(est, X[:, :2], labels=y, output_path=outdir / "support_ellipse.png", show=False)
    rc.plot_robust_distance_profile(est, labels=y, output_path=outdir / "distance_profile.png", show=False)
    print("saved diagnostics to", outdir)
