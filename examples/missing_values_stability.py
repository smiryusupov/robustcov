"""Stress test robust estimators with increasing missingness.

The MVP handles NaNs with deterministic column-median imputation when
missing_values='median'. This is deliberately simple: the package should be
robust-estimation focused, not a full missing-data package.

Run:
    python examples/missing_values_stability.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def relative_fro_error(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def inject_missing(X, rate, rng):
    X = X.copy()
    mask = rng.random(X.shape) < rate
    X[mask] = np.nan
    return X


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n, p = 700, 6
    Sigma = 0.6 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    X[:70] += rng.normal(7.0, 1.0, size=(70, p))  # 10% contamination

    print("relative covariance error under contamination + missing values")
    print("missing_rate, empirical_median_impute, FastMCD, RegTyler")
    for rate in [0.00, 0.05, 0.10, 0.20, 0.30]:
        Xm = inject_missing(X, rate, rng)
        Xi = rc.RobustMedianImputer().fit_transform(Xm)
        empirical = np.cov(Xi, rowvar=False)
        mcd = rc.FastMCD(quality="balanced", random_state=1, missing_values="median").fit(Xm).covariance_
        tyler = rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median", missing_values="median").fit(Xm).covariance_
        print(
            f"{rate:.2f}, "
            f"{relative_fro_error(empirical, Sigma):.3f}, "
            f"{relative_fro_error(mcd, Sigma):.3f}, "
            f"{relative_fro_error(tyler, Sigma):.3f}"
        )
