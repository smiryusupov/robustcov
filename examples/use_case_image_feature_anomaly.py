"""Use case: image-feature anomaly detection with sklearn digits.

The example treats one digit as normal and another digit as anomaly. PCA features
are fed to robustcov estimators and robust distances become anomaly scores.

Run:
    python examples/use_case_image_feature_anomaly.py
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
    try:
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise SystemExit(f"This example requires scikit-learn: {exc}")

    normal_digit, anomaly_digit = 0, 1
    anomaly_count = 30
    rng = np.random.default_rng(0)
    digits = load_digits()
    normal_idx = np.where(digits.target == normal_digit)[0]
    anomaly_idx = np.where(digits.target == anomaly_digit)[0]
    anomaly_idx = rng.choice(anomaly_idx, size=anomaly_count, replace=False)
    idx = np.r_[normal_idx, anomaly_idx]
    X_raw = digits.data[idx]
    labels = np.r_[np.zeros(normal_idx.size, dtype=int), np.ones(anomaly_idx.size, dtype=int)]
    X = StandardScaler().fit_transform(X_raw)
    X = PCA(n_components=12, random_state=0).fit_transform(X)

    est = rc.FastMCD(quality="fast", contamination=float(labels.mean()), random_state=0).fit(X)
    det = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("image-feature one-class anomaly example")
    print(f"normal_digit={normal_digit}, anomaly_digit={anomaly_digit}, anomaly_count={anomaly_count}, n={X.shape[0]}, p={X.shape[1]}")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/image_features")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
