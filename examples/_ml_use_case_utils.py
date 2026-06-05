"""Small helper functions for robustcov ML use-case examples."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def binary_metrics(y_true, pred_outlier, scores=None):
    y = np.asarray(y_true, dtype=bool)
    pred = np.asarray(pred_outlier, dtype=bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = dict(precision=precision, recall=recall, f1=f1, detected=int(pred.sum()), tp=tp, fp=fp, fn=fn, tn=tn)
    if scores is not None:
        try:
            from sklearn.metrics import roc_auc_score
            out['roc_auc'] = float(roc_auc_score(y.astype(int), scores))
        except Exception:
            out['roc_auc'] = float('nan')
    else:
        out['roc_auc'] = float('nan')
    return out


def print_rows(title, rows):
    fields = ['method', 'seconds', 'precision', 'recall', 'f1', 'roc_auc', 'detected']
    print(title)
    print(','.join(fields))
    for r in rows:
        vals = []
        for f in fields:
            v = r.get(f, '')
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        print(','.join(vals))


def save_rows_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['method', 'seconds', 'precision', 'recall', 'f1', 'roc_auc', 'detected']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def plot_metric_bars(rows, metric, output_path, title=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r['method'].replace('sklearn ', '').replace('robustcov ', '') for r in rows]
    vals = [float(r.get(metric, 0.0)) for r in rows]
    fig = plt.figure(figsize=(8, 4.2))
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.set_ylim(0, max(1.0, max(vals) * 1.15 if vals else 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_score_profile(scores, labels, output_path, title='Anomaly score profile'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores)
    y = scores[order]
    lab = labels[order]
    fig = plt.figure(figsize=(8, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(y) + 1), y, marker='o', markersize=2, linewidth=1)
    idx = np.where(lab)[0]
    if idx.size:
        ax.scatter(idx + 1, y[idx], s=28, facecolors='none', edgecolors='black', label='known anomaly')
        ax.legend()
    ax.set_xlabel('Observation rank by anomaly score')
    ax.set_ylabel('Anomaly score')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def baseline_rows(X, labels, contamination, robust_estimator='fastmcd'):
    import time
    import robustcov as rc
    rows = []
    alpha = 1.0 - float(contamination)

    def add(method, pred, scores, seconds):
        row = {'method': method, 'seconds': seconds}
        row.update(binary_metrics(labels, pred, scores))
        rows.append(row)

    t0 = time.perf_counter()
    if robust_estimator == 'auto':
        det = rc.AutoRobustAnomalyDetector(contamination=contamination).fit(X)
        add('robustcov Auto', det.labels_ == -1, det.score_, time.perf_counter() - t0)
        primary = det
    else:
        est = rc.FastMCD(quality='fast', contamination=min(float(contamination), 0.49), random_state=0, n_jobs=1).fit(X)
        det = rc.RobustOutlierDetector(estimator=est, threshold='empirical', alpha=alpha).fit(X)
        add('robustcov FastMCD', det.labels_ == -1, det.distances_, time.perf_counter() - t0)
        primary = est

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
        from sklearn.covariance import EllipticEnvelope

        t0 = time.perf_counter()
        clf = IsolationForest(contamination=contamination, random_state=0).fit(X)
        add('sklearn IsolationForest', clf.predict(X) == -1, -clf.score_samples(X), time.perf_counter() - t0)

        t0 = time.perf_counter()
        clf = LocalOutlierFactor(contamination=contamination, novelty=False)
        pred = clf.fit_predict(X) == -1
        add('sklearn LocalOutlierFactor', pred, -clf.negative_outlier_factor_, time.perf_counter() - t0)

        t0 = time.perf_counter()
        clf = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale').fit(X)
        add('sklearn OneClassSVM', clf.predict(X) == -1, -clf.decision_function(X).ravel(), time.perf_counter() - t0)

        t0 = time.perf_counter()
        clf = EllipticEnvelope(contamination=contamination, random_state=0).fit(X)
        add('sklearn EllipticEnvelope', clf.predict(X) == -1, -clf.decision_function(X), time.perf_counter() - t0)
    except Exception as exc:
        rows.append({'method': f'sklearn baselines unavailable: {exc}', 'seconds': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'detected': 0})
    rows.sort(key=lambda r: (-float(r.get('f1', 0.0)), float(r.get('seconds', 0.0))))
    return rows, primary
