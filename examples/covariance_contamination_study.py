"""Show covariance error as contamination increases.

The point is to demonstrate the package positioning: efficient robust estimators
stay closer to the clean covariance than empirical covariance under outliers.

Run:
    python examples/covariance_contamination_study.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def true_covariance(p):
    return 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))


def relative_fro_error(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def contaminate(X, fraction, rng):
    X = X.copy()
    n, p = X.shape
    m = int(round(fraction * n))
    if m:
        idx = rng.choice(n, size=m, replace=False)
        X[idx] += rng.normal(loc=8.0, scale=1.5, size=(m, p))
    return X


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p = 800, 8
    Sigma = true_covariance(p)
    X_clean = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

    estimators = {
        "Empirical": lambda X: np.cov(X, rowvar=False),
        "FastMCD": lambda X: rc.FastMCD(quality="balanced", random_state=1).fit(X).covariance_,
        "RegTyler": lambda X: rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median").fit(X).covariance_,
    }

    print("relative covariance error vs true clean covariance")
    print("contamination, " + ", ".join(estimators))
    for frac in [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]:
        X = contaminate(X_clean, frac, rng)
        row = [f"{frac:.2f}"]
        for fit in estimators.values():
            cov = fit(X)
            row.append(f"{relative_fro_error(cov, Sigma):.3f}")
        print(", ".join(row))
