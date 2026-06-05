"""Real ML anomaly-detection smoke test using sklearn datasets when available.

This is not a definitive benchmark. It is a practical example showing how robustcov can
be used as a robust-distance anomaly detector on a real tabular dataset.

Run:
    python examples/real_ml_anomaly.py
"""
from __future__ import annotations

import time

import numpy as np
import robustcov as rc


def metrics(y_true, y_pred_anomaly):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred_anomaly, dtype=bool)
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, int(np.sum(y_pred))


if __name__ == '__main__':
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        from sklearn.covariance import MinCovDet
        from sklearn.ensemble import IsolationForest
    except Exception as exc:
        raise SystemExit(f'scikit-learn is required for this example: {exc}')

    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    # Treat malignant cases as the target anomaly class for this demo.
    y_anom = data.target == 0
    contamination = float(np.mean(y_anom))

    methods = []
    methods.append(('robustcov FastMCD', lambda: rc.RobustOutlierDetector(
        estimator=rc.FastMCD(quality='balanced', contamination=min(contamination, 0.49), random_state=0),
        threshold='empirical',
        alpha=1.0 - contamination,
    )))
    methods.append(('sklearn MinCovDet', lambda: MinCovDet(random_state=0, support_fraction=max(0.51, 1.0 - contamination))))
    methods.append(('sklearn IsolationForest', lambda: IsolationForest(contamination=contamination, random_state=0)))

    print('dataset: sklearn breast_cancer')
    print('n,p,anomaly_fraction:', X.shape[0], X.shape[1], round(contamination, 4))
    print('method,seconds,precision,recall,detected')

    for name, factory in methods:
        t0 = time.perf_counter()
        model = factory()
        if name == 'robustcov FastMCD':
            labels = model.fit_predict(X)
            pred_anom = labels == -1
        elif name == 'sklearn MinCovDet':
            est = model.fit(X)
            d = est.mahalanobis(X)
            cutoff = np.quantile(d, 1.0 - contamination)
            pred_anom = d > cutoff
        else:
            labels = model.fit_predict(X)
            pred_anom = labels == -1
        seconds = time.perf_counter() - t0
        precision, recall, detected = metrics(y_anom, pred_anom)
        print(f'{name},{seconds:.6f},{precision:.4f},{recall:.4f},{detected}')
