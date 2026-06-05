"""Compare diagnostic and stability AutoRobustScatter selection.

Run:
    python examples/auto_selection_stability.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


if __name__ == "__main__":
    rng = np.random.default_rng(321)
    n, p, df = 30, 80, 1.0
    Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    g = rng.chisquare(df, size=n) / df
    X = Z / np.sqrt(g)[:, None]

    for selection in ["diagnostic", "stability"]:
        auto = rc.AutoRobustScatter(selection=selection, n_splits=3, random_state=0).fit(X)
        print()
        print("selection:", selection)
        print(auto.summary())
        print("rel_fro_error:", f"{rel_fro(auto.covariance_, Sigma):.4f}")
