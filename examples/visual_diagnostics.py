"""Generate visual diagnostics for a robust estimator.

Run:
    python examples/visual_diagnostics.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


if __name__ == "__main__":
    outdir = Path('results/visual_diagnostics')
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    p = 5
    Sigma = 0.6 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=500)
    X[:40] += rng.normal(7.0, 1.0, size=(40, p))

    est = rc.FastMCD(quality='balanced', random_state=0).fit(X)

    rc.plot_mahalanobis_diagnostics(est, output_path=outdir / 'mahalanobis_diagnostics.png', show=False)
    rc.plot_mahalanobis_qq(est, output_path=outdir / 'mahalanobis_qq.png', show=False)
    rc.plot_distance_histogram(est, output_path=outdir / 'distance_histogram.png', show=False)
    rc.plot_covariance_heatmap(est.covariance_, title='FastMCD covariance', output_path=outdir / 'covariance_heatmap.png', show=False)

    print('saved visual diagnostics to', outdir)
    print('radial_kurtosis:', getattr(est, 'radial_kurtosis_', None))
    print('support size:', int(est.support_.sum()))
