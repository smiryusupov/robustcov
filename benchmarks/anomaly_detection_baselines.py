"""Compare robustcov anomaly detectors with common sklearn anomaly baselines.

The benchmark is intentionally small and reproducible. It is designed for the
Sphinx benchmark gallery rather than for publication claims.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import robustcov as rc


def make_data(n=900, p=12, contamination=0.08, random_state=0):
    rng = np.random.default_rng(random_state)
    n_out = int(round(n * contamination))
    n_in = n - n_out
    # Heavy-tailed but structured inliers.
    A = rng.normal(size=(p, p))
    cov = A @ A.T / p + 0.30 * np.eye(p)
    z = rng.multivariate_normal(np.zeros(p), cov, size=n_in)
    scale = np.sqrt(rng.chisquare(df=3, size=n_in) / 3.0)
    X_in = z / scale[:, None]
    # Mixed outliers: some mean-shift, some variance/leverage.
    X_out = rng.multivariate_normal(np.ones(p) * 4.0, np.eye(p), size=n_out)
    if n_out > 4:
        X_out[: n_out // 2] += rng.normal(scale=4.0, size=(n_out // 2, p))
    X = np.vstack([X_in, X_out])
    y = np.r_[np.zeros(n_in, dtype=int), np.ones(n_out, dtype=int)]
    perm = rng.permutation(n)
    return X[perm], y[perm]


def metrics(y_true, labels_outlier, scores=None):
    pred = np.asarray(labels_outlier, dtype=bool)
    y = np.asarray(y_true, dtype=bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {"precision": precision, "recall": recall, "f1": f1, "detected": int(np.sum(pred))}
    if scores is not None:
        try:
            from sklearn.metrics import roc_auc_score
            out["roc_auc"] = float(roc_auc_score(y.astype(int), scores))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def run_one(name, fit_predict_fn, y):
    t0 = time.perf_counter()
    labels_out, scores = fit_predict_fn()
    seconds = time.perf_counter() - t0
    row = {"method": name, "seconds": seconds}
    row.update(metrics(y, labels_out, scores))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--contamination", type=float, default=0.08)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    X, y = make_data(args.n, args.p, args.contamination, args.random_state)
    alpha = 1.0 - args.contamination
    rows = []

    def robustcov_fastmcd():
        det = rc.RobustOutlierDetector(
            estimator=rc.FastMCD(random_state=0, contamination=args.contamination, quality="balanced"),
            threshold="empirical",
            alpha=alpha,
        ).fit(X)
        return det.labels_ == -1, det.distances_

    rows.append(run_one("robustcov FastMCD detector", robustcov_fastmcd, y))

    def robustcov_auto():
        det = rc.AutoRobustAnomalyDetector(contamination=args.contamination).fit(X)
        return det.labels_ == -1, det.score_

    rows.append(run_one("robustcov Auto detector", robustcov_auto, y))

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
        from sklearn.covariance import EllipticEnvelope

        def iso():
            clf = IsolationForest(contamination=args.contamination, random_state=0).fit(X)
            labels = clf.predict(X) == -1
            scores = -clf.score_samples(X)
            return labels, scores
        rows.append(run_one("sklearn IsolationForest", iso, y))

        def lof():
            clf = LocalOutlierFactor(contamination=args.contamination, novelty=False)
            labels = clf.fit_predict(X) == -1
            scores = -clf.negative_outlier_factor_
            return labels, scores
        rows.append(run_one("sklearn LocalOutlierFactor", lof, y))

        def ocsvm():
            clf = OneClassSVM(nu=args.contamination, kernel="rbf", gamma="scale").fit(X)
            labels = clf.predict(X) == -1
            scores = -clf.decision_function(X).ravel()
            return labels, scores
        rows.append(run_one("sklearn OneClassSVM", ocsvm, y))

        def elliptic():
            clf = EllipticEnvelope(contamination=args.contamination, random_state=0).fit(X)
            labels = clf.predict(X) == -1
            scores = -clf.decision_function(X)
            return labels, scores
        rows.append(run_one("sklearn EllipticEnvelope", elliptic, y))
    except Exception as exc:
        rows.append({"method": f"sklearn baselines unavailable: {exc}", "seconds": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0, "detected": 0})

    rows.sort(key=lambda r: (-float(r.get("f1", 0)), float(r.get("seconds", 0))))
    fields = ["method", "seconds", "precision", "recall", "f1", "roc_auc", "detected"]
    print(f"n={args.n}, p={args.p}, contamination={args.contamination}")
    print(",".join(fields))
    for r in rows:
        print(",".join(str(round(r[c], 6)) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in fields))

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
