"""Use case: multivariate sensor anomaly detection.

This synthetic example imitates correlated sensor readings with a small burst of abnormal
readings. Robust distances expose the burst and the distance profile makes the threshold
behavior visible.

Run:
    python examples/use_case_sensor_anomaly.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    n, p = 600, 6
    Sigma = 0.55 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    labels = np.zeros(n, dtype=int)
    burst = np.arange(260, 295)
    labels[burst] = 1
    X[burst, :3] += rng.normal(5.0, 0.6, size=(burst.size, 3))
    X[burst, 3:] += rng.normal(-3.5, 0.6, size=(burst.size, 3))

    est = rc.FastMCD(quality="balanced", random_state=0).fit(X)
    detector = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = detector.labels_ == -1
    precision = np.sum(pred & (labels == 1)) / max(1, np.sum(pred))
    recall = np.sum(pred & (labels == 1)) / max(1, np.sum(labels == 1))

    print("sensor anomaly use case")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}, support={int(est.support_.sum())}")

    outdir = Path("results/use_cases/sensor")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_profile(est, labels=labels, output_path=outdir / "distance_profile.png", show=False)
    rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
