"""Use case: biomedical / signal-window covariance.

This example simulates multichannel signal-window features. A few windows contain
abnormal high-energy bursts. Robust distances find windows whose covariance
feature pattern is unusual.

Run:
    python examples/use_case_biomedical_signal.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def window_features(windows):
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    energy = np.mean(windows ** 2, axis=1)
    peak = np.max(np.abs(windows), axis=1)
    return np.hstack([mean, std, energy, peak])


def precision_recall(pred, labels):
    pred = np.asarray(pred, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    return (np.sum(pred & labels) / max(1, np.sum(pred)),
            np.sum(pred & labels) / max(1, np.sum(labels)))


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n_windows, length, channels = 500, 80, 4
    t = np.linspace(0, 2 * np.pi, length)
    base = np.stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)], axis=1)
    windows = base[None, :, :] + 0.25 * rng.normal(size=(n_windows, length, channels))
    labels = np.zeros(n_windows, dtype=int)
    idx = rng.choice(n_windows, size=35, replace=False)
    labels[idx] = 1
    windows[idx, 20:45, :2] += rng.normal(2.5, 0.4, size=(idx.size, 25, 2))
    windows[idx, 50:65, 2:] -= rng.normal(2.0, 0.4, size=(idx.size, 15, 2))

    X = window_features(windows)
    est = rc.FastMCD(quality="balanced", random_state=2).fit(X)
    det = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("biomedical/signal-window anomaly example")
    print(f"features={X.shape[1]}, precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/biomedical_signal")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_profile(est, labels=labels, output_path=outdir / "distance_profile.png", show=False)
    print("saved diagnostics to", outdir)
