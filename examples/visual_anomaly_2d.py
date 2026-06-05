"""2D anomaly diagnostic plot with robust covariance ellipse.

Run:
    python examples/visual_anomaly_2d.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


if __name__ == "__main__":
    outdir = Path('results/visual_anomaly_2d')
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    cov = np.array([[1.0, 0.75], [0.75, 1.4]])
    X_clean = rng.multivariate_normal([0.0, 0.0], cov, size=450)
    X_out = rng.multivariate_normal([5.5, 5.0], [[0.5, 0.1], [0.1, 0.5]], size=50)
    X = np.vstack([X_clean, X_out])
    y = np.r_[np.zeros(len(X_clean), dtype=int), np.ones(len(X_out), dtype=int)]

    est = rc.FastMCD(quality='balanced', random_state=0).fit(X)
    rc.plot_anomaly_scatter_2d(est, X, labels=y, output_path=outdir / 'support_ellipse.png', show=False)
    rc.plot_distance_scatter_2d(est, X, output_path=outdir / 'distance_colored.png', show=False)
    rc.plot_mahalanobis_qq(est, output_path=outdir / 'qq.png', show=False)

    print('saved 2D anomaly diagnostics to', outdir)
    print('support size:', int(est.support_.sum()))
    print('radial_kurtosis:', round(est.radial_kurtosis_, 3))
