"""One-class anomaly smoke test on sklearn digits.

The script treats one digit as normal and one digit as anomaly. It is not intended
as a formal benchmark; it is a compact real-ML diagnostic example.

Run:
    python examples/real_ml_one_class_digits.py --normal-digit 1 --anomaly-digit 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import robustcov as rc


def precision_recall(y_true, labels):
    pred = labels == -1
    tp = int(np.sum(pred & y_true))
    fp = int(np.sum(pred & ~y_true))
    fn = int(np.sum((~pred) & y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, int(np.sum(pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-digit", type=int, default=1)
    parser.add_argument("--anomaly-digit", type=int, default=7)
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    try:
        from sklearn.datasets import load_digits
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.covariance import MinCovDet
        from sklearn.ensemble import IsolationForest
    except Exception as exc:
        raise SystemExit(f"This example requires scikit-learn: {exc}")

    data = load_digits()
    mask = (data.target == args.normal_digit) | (data.target == args.anomaly_digit)
    X = data.data[mask]
    y = data.target[mask]
    y_true = y == args.anomaly_digit
    X = StandardScaler().fit_transform(X)

    # Keep a compact but nontrivial feature space. This also makes covariance methods
    # more stable for a small one-class example.
    X_model = PCA(n_components=min(12, X.shape[1]), random_state=0).fit_transform(X)

    methods = {
        "robustcov FastMCD": lambda: rc.RobustOutlierDetector(
            rc.FastMCD(quality="balanced", random_state=0, contamination=args.contamination),
            threshold="empirical",
            alpha=1.0 - args.contamination,
        ),
        "robustcov Auto": lambda: rc.AutoRobustAnomalyDetector(contamination=args.contamination),
        "sklearn MinCovDet": lambda: None,
        "sklearn IsolationForest": lambda: IsolationForest(contamination=args.contamination, random_state=0),
    }

    print(f"dataset: sklearn digits normal={args.normal_digit}, anomaly={args.anomaly_digit}")
    print("n,p,anomaly_fraction:", X_model.shape[0], X_model.shape[1], f"{float(y_true.mean()):.4f}")
    print("method,seconds,precision,recall,detected")

    for name, factory in methods.items():
        t0 = time.perf_counter()
        if name == "sklearn MinCovDet":
            est = MinCovDet(random_state=0).fit(X_model)
            d = est.mahalanobis(X_model)
            threshold = np.quantile(d, 1.0 - args.contamination)
            labels = np.where(d <= threshold, 1, -1)
        elif name == "sklearn IsolationForest":
            est = factory().fit(X_model)
            labels = est.predict(X_model)
        else:
            est = factory().fit(X_model)
            labels = est.labels_
        seconds = time.perf_counter() - t0
        precision, recall, detected = precision_recall(y_true, labels)
        print(f"{name},{seconds:.6f},{precision:.4f},{recall:.4f},{detected}")

    if args.plot:
        outdir = Path("results/one_class_digits")
        outdir.mkdir(parents=True, exist_ok=True)
        X_2d = PCA(n_components=2, random_state=0).fit_transform(X)
        est2d = rc.FastMCD(quality="balanced", random_state=0, contamination=args.contamination).fit(X_2d)
        rc.plot_anomaly_scatter_2d(est2d, X_2d, labels=y_true, output_path=outdir / "digits_2d.png", show=False)
        rc.plot_distance_scatter_2d(est2d, X_2d, output_path=outdir / "digits_distance.png", show=False)
        print("saved plots to", outdir)
