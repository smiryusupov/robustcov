"""Show how contamination-aware support selection affects FastMCD.

This example is intentionally simple: it compares the default MCD support size
against a user-provided contamination estimate. In real data, contamination is
usually approximate, so treat it as a tuning knob rather than a ground truth.
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


rng = np.random.default_rng(42)
n, p = 1000, 8
true_contamination = 0.30
Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
idx = rng.choice(n, size=int(true_contamination * n), replace=False)
X[idx] += rng.normal(8.0, 1.5, size=(len(idx), p))

estimators = {
    "default h": rc.FastMCD(quality="balanced", random_state=0),
    "contamination-aware h": rc.FastMCD(quality="balanced", contamination=true_contamination, random_state=0),
}

print("method, h_raw, support_size, rel_fro_error")
for name, est in estimators.items():
    est.fit(X)
    print(f"{name}, {est.h_}, {int(est.support_.sum())}, {rel_fro(est.covariance_, Sigma):.4f}")
