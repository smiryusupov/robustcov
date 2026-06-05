"""Anomaly detection with robust covariance estimators.

Run:
    python examples/anomaly_detection.py
"""
from __future__ import annotations

import numpy as np
import robustcov as rc


def make_data(n_clean=450, n_outliers=50, p=5, seed=0):
    rng = np.random.default_rng(seed)
    A = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X_clean = rng.multivariate_normal(np.zeros(p), A, size=n_clean)
    X_out = rng.multivariate_normal(np.full(p, 7.0), 0.4 * np.eye(p), size=n_outliers)
    X = np.vstack([X_clean, X_out])
    y = np.r_[np.ones(n_clean, dtype=int), -np.ones(n_outliers, dtype=int)]
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def metrics(y_true, y_pred):
    out_true = y_true == -1
    out_pred = y_pred == -1
    tp = np.sum(out_true & out_pred)
    fp = np.sum(~out_true & out_pred)
    fn = np.sum(out_true & ~out_pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, int(out_pred.sum())


if __name__ == "__main__":
    X, y = make_data()

    detectors = {
        "FastMCD + empirical threshold": rc.RobustOutlierDetector(
            estimator=rc.FastMCD(quality="balanced", random_state=0),
            threshold="empirical",
            alpha=0.90,
        ),
        "RegularizedTyler + empirical threshold": rc.RobustOutlierDetector(
            estimator=rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median"),
            threshold="empirical",
            alpha=0.90,
        ),
    }

    print("anomaly detection example")
    print("method, precision, recall, detected, radial_kurtosis")
    for name, det in detectors.items():
        pred = det.fit_predict(X)
        precision, recall, detected = metrics(y, pred)
        print(f"{name}, {precision:.3f}, {recall:.3f}, {detected}, {det.estimator_.radial_kurtosis_:.3f}")
