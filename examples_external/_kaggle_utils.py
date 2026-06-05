"""Utilities shared by optional Kaggle/external examples.

These helpers are deliberately outside the package test path. They require pandas
and scikit-learn only when a user chooses to run external examples.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np


def ensure_deps():
    try:
        import pandas as pd  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional path
        raise SystemExit(
            "External examples require pandas and scikit-learn. Install with:\n"
            "  python -m pip install pandas scikit-learn\n"
            f"Original import error: {exc}"
        )


def read_csv(path, max_rows=None):
    ensure_deps()
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")
    return pd.read_csv(path, nrows=max_rows)


def find_label_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def numeric_matrix(df, label_col=None, drop_cols=(), max_categories=20):
    import pandas as pd
    cols_to_drop = [c for c in [label_col, *drop_cols] if c and c in df.columns]
    Xdf = df.drop(columns=cols_to_drop, errors='ignore').copy()
    # Keep numeric columns and small-cardinality categoricals encoded as dummies.
    numeric = Xdf.select_dtypes(include=['number', 'bool'])
    cats = []
    for col in Xdf.select_dtypes(exclude=['number', 'bool']).columns:
        if Xdf[col].nunique(dropna=True) <= max_categories:
            cats.append(col)
    if cats:
        encoded = pd.get_dummies(Xdf[cats], dummy_na=True)
        numeric = pd.concat([numeric, encoded], axis=1)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    med = numeric.median(numeric_only=True)
    numeric = numeric.fillna(med).fillna(0.0)
    return numeric.to_numpy(dtype=float), list(numeric.columns)


def scale_matrix(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def binary_metrics(y_true, pred_outlier, scores=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred_outlier, dtype=int)
    out = {
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'detected': int(pred.sum()),
    }
    if scores is not None and len(np.unique(y)) == 2:
        out['roc_auc'] = float(roc_auc_score(y, scores))
        out['pr_auc'] = float(average_precision_score(y, scores))
    else:
        out['roc_auc'] = float('nan')
        out['pr_auc'] = float('nan')
    return out


def evaluate_external_baselines(X, y, contamination=0.02, include_slow=False, robust='auto'):
    """Evaluate robustcov and familiar sklearn anomaly baselines."""
    import robustcov as rc
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope

    rows = []

    def add(method, pred, scores, seconds):
        row = {'method': method, 'seconds': float(seconds)}
        row.update(binary_metrics(y, pred, scores))
        rows.append(row)

    t0 = time.perf_counter()
    if robust == 'auto':
        auto = rc.AutoRobustScatter(selection='diagnostic', random_state=0).fit(X)
        est = auto.estimator_
        name = f"robustcov Auto({auto.best_estimator_name_})"
    elif robust == 'cauchy':
        est = rc.RegularizedCauchy(alpha=0.10, warn_on_nonconvergence=False).fit(X)
        name = 'robustcov RegularizedCauchy'
    else:
        est = rc.FastMCD(quality='fast', contamination=min(contamination, 0.49), random_state=0, n_jobs=1).fit(X)
        name = 'robustcov FastMCD'
    det = rc.RobustOutlierDetector(estimator=est, threshold='empirical', alpha=1.0 - contamination).fit(X)
    add(name, det.labels_ == -1, det.distances_, time.perf_counter() - t0)
    primary_scores = np.asarray(det.distances_, dtype=float)

    t0 = time.perf_counter()
    iso = IsolationForest(contamination=contamination, random_state=0, n_estimators=200).fit(X)
    add('sklearn IsolationForest', iso.predict(X) == -1, -iso.score_samples(X), time.perf_counter() - t0)

    t0 = time.perf_counter()
    lof = LocalOutlierFactor(contamination=contamination)
    pred = lof.fit_predict(X) == -1
    add('sklearn LocalOutlierFactor', pred, -lof.negative_outlier_factor_, time.perf_counter() - t0)

    if include_slow:
        t0 = time.perf_counter()
        svm = OneClassSVM(nu=max(contamination, 1e-3), gamma='scale').fit(X)
        add('sklearn OneClassSVM', svm.predict(X) == -1, -svm.decision_function(X).ravel(), time.perf_counter() - t0)

    try:
        t0 = time.perf_counter()
        ell = EllipticEnvelope(contamination=contamination, random_state=0).fit(X)
        add('sklearn EllipticEnvelope', ell.predict(X) == -1, -ell.decision_function(X), time.perf_counter() - t0)
    except Exception as exc:
        rows.append({'method': f'sklearn EllipticEnvelope failed: {type(exc).__name__}', 'seconds': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0, 'detected': 0})

    rows.sort(key=lambda r: (-r.get('pr_auc', 0.0), -r.get('f1', 0.0)))
    return rows, primary_scores


def save_table(rows, path):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def print_table(rows, title):
    fields = ['method', 'seconds', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'detected']
    print(title)
    print(','.join(fields))
    for row in rows:
        vals = []
        for f in fields:
            v = row.get(f, '')
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        print(','.join(vals))


def plot_metric(rows, metric, output_path, title):
    import matplotlib.pyplot as plt
    names = [r['method'].replace('sklearn ', '').replace('robustcov ', '') for r in rows]
    vals = [float(r.get(metric, 0.0)) for r in rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(vals) * 1.15 if vals else 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_score_profile(scores, y, output_path, title):
    import matplotlib.pyplot as plt
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=bool)
    order = np.argsort(scores)
    xs = np.arange(1, len(scores) + 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(xs, scores[order], linewidth=1.2)
    pos = np.where(y[order])[0]
    if pos.size:
        ax.scatter(pos + 1, scores[order][pos], s=20, facecolors='none', edgecolors='black', label='positive class')
        ax.legend()
    ax.set_xlabel('Observation rank by robust score')
    ax.set_ylabel('Robust anomaly score')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_markdown_summary(path, title, rows, notes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'# {title}', '', *notes, '', '## Metrics', '', '| method | precision | recall | F1 | ROC AUC | PR AUC | seconds |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | {r.get('f1', 0):.3f} | {r.get('roc_auc', 0):.3f} | {r.get('pr_auc', 0):.3f} | {r.get('seconds', 0):.3f} |")
    path.write_text('\n'.join(lines) + '\n')



def write_external_registry(rows, outdir):
    """Write a small external-results registry as CSV, Markdown, and HTML."""
    ensure_deps()
    import pandas as pd
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=['dataset', 'task', 'best_method', 'best_pr_auc', 'best_f1', 'best_seconds', 'path'])
    df.to_csv(outdir / 'external_results.csv', index=False)

    md = ['# External results registry', '', '| dataset | task | best method | PR AUC | F1 | seconds | path |', '|---|---|---|---:|---:|---:|---|']
    for _, r in df.iterrows():
        md.append(f"| {r.get('dataset','')} | {r.get('task','')} | {r.get('best_method','')} | {float(r.get('best_pr_auc', float('nan'))):.3f} | {float(r.get('best_f1', float('nan'))):.3f} | {float(r.get('best_seconds', float('nan'))):.3f} | `{r.get('path','')}` |")
    (outdir / 'external_results.md').write_text('\n'.join(md) + '\n')

    html = ['<!doctype html><html><head><meta charset="utf-8"><title>robustcov external results</title>',
            '<style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px} th{background:#f5f5f5} code{background:#f6f6f6;padding:2px 4px;border-radius:4px}</style>',
            '</head><body><h1>robustcov external results registry</h1>',
            '<p>Optional external/Kaggle runs. These results are not part of the core test suite.</p>',
            df.to_html(index=False, escape=False), '</body></html>']
    (outdir / 'external_results.html').write_text('\n'.join(html))
