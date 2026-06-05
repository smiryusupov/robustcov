"""Use case: fraud-style tabular anomaly screening.

This synthetic example imitates transaction features such as amount, velocity,
merchant diversity, device age, and geo-change. Robust distances are used as a
first-pass triage score.

Run:
    python examples/use_case_fraud_screening.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def precision_recall(pred, labels):
    pred = np.asarray(pred, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    precision = np.sum(pred & labels) / max(1, np.sum(pred))
    recall = np.sum(pred & labels) / max(1, np.sum(labels))
    return precision, recall


if __name__ == "__main__":
    rng = np.random.default_rng(21)
    n, p = 1200, 8
    Sigma = 0.45 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    labels = np.zeros(n, dtype=int)
    k = 70
    idx = rng.choice(n, size=k, replace=False)
    labels[idx] = 1
    # high amount, high velocity, unusual merchant/device/geography mix
    X[idx, 0] += rng.normal(6.0, 0.8, size=k)
    X[idx, 1] += rng.normal(4.0, 0.7, size=k)
    X[idx, 2] -= rng.normal(3.5, 0.6, size=k)
    X[idx, 5:] += rng.normal(2.8, 0.8, size=(k, 3))

    est = rc.FastMCD(quality="balanced", random_state=0).fit(X)
    det = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("fraud-style tabular anomaly screening")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}, support={int(est.support_.sum())}")

    outdir = Path("results/use_cases/fraud")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_profile(est, labels=labels, output_path=outdir / "distance_profile.png", show=False)
    rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
