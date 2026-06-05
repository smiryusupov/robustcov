"""Use case: predictive-maintenance monitoring.

Vibration/temperature/load features drift before failures. This synthetic example
uses robust distances to flag abnormal machine-state windows.

Run:
    python examples/use_case_maintenance_monitoring.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def precision_recall(pred, labels):
    pred = np.asarray(pred, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    return (np.sum(pred & labels) / max(1, np.sum(pred)),
            np.sum(pred & labels) / max(1, np.sum(labels)))


if __name__ == "__main__":
    rng = np.random.default_rng(27)
    n, p = 720, 9
    Sigma = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    labels = np.zeros(n, dtype=int)
    # gradual pre-failure windows
    idx = np.r_[np.arange(240, 275), np.arange(520, 555)]
    labels[idx] = 1
    ramp = np.linspace(0, 1, idx.size)[:, None]
    X[idx, :3] += 2.0 + 3.0 * ramp
    X[idx, 3:6] += rng.normal(2.0, 0.5, size=(idx.size, 3))

    est = rc.FastMCD(quality="fast", contamination=float(labels.mean()), random_state=1).fit(X)
    det = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("predictive-maintenance monitoring")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/maintenance")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_profile(est, labels=labels, sort=False, output_path=outdir / "time_profile.png", show=False)
    rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
