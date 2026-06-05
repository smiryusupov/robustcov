"""AutoRobustScatter demo.

Run:
    python examples/auto_robust_scatter.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n, p, df = 45, 60, 1.5
    Sigma = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    g = rng.chisquare(df, size=n) / df
    X = Z / np.sqrt(g)[:, None]

    auto = rc.AutoRobustScatter(selection="stability", n_splits=3, random_state=0).fit(X)
    print(auto.summary())
    print("selected rel_fro_error:", f"{rel_fro(auto.covariance_, Sigma):.4f}")
    print("diagnostic report:")
    print(auto.report().summary())
