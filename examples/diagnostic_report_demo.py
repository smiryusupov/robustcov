"""Show the robust diagnostic report.

Run:
    python examples/diagnostic_report_demo.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    p = 6
    Sigma = 0.6 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=600)
    X[:80] += rng.normal(7.0, 1.2, size=(80, p))

    est = rc.FastMCD(quality="balanced", random_state=0).fit(X)
    report = rc.diagnostic_report(est, alpha=0.975)
    print(report.summary())
