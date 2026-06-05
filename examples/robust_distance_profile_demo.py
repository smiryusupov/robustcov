"""Robust distance profile/proline-style diagnostic demo.

Run:
    python examples/robust_distance_profile_demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


if __name__ == '__main__':
    outdir = Path('results/distance_profile')
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    p = 6
    Sigma = 0.6 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=500)
    y = np.zeros(500, dtype=int)
    idx = rng.choice(500, size=45, replace=False)
    X[idx] += rng.normal(7.0, 1.0, size=(45, p))
    y[idx] = 1

    est = rc.FastMCD(quality='balanced', random_state=0).fit(X)
    rc.plot_robust_distance_profile(est, labels=y, output_path=outdir / 'distance_profile.png', show=False)
    rc.plot_robust_distance_panel(est, output_path=outdir / 'distance_panel.png', show=False)
    rc.plot_mahalanobis_qq(est, output_path=outdir / 'qq.png', show=False)

    print('saved robust distance diagnostics to', outdir)
    print('radial_kurtosis:', round(est.radial_kurtosis_, 3))
    print('support size:', int(est.support_.sum()))
