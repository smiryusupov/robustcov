#!/usr/bin/env python
"""Optional external example: medical tabular screening with robust distances."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from _kaggle_utils import read_csv, find_label_column, numeric_matrix, scale_matrix, evaluate_external_baselines, save_table, print_table, plot_metric, plot_score_profile, write_markdown_summary


def to_binary(series, positive=None):
    if positive is not None:
        return (series.astype(str).str.lower() == str(positive).lower()).astype(int).to_numpy()
    if series.dtype.kind in 'biufc':
        vals = series.to_numpy()
        uniq = np.unique(vals[~np.isnan(vals)] if vals.dtype.kind == 'f' else vals)
        if len(uniq) == 2:
            return (vals == max(uniq)).astype(int)
    s = series.astype(str).str.lower()
    positives = {'m', 'malignant', 'positive', 'disease', '1', 'true', 'yes'}
    return s.isin(positives).astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser(description='Medical tabular screening with robustcov')
    ap.add_argument('--data', required=True, help='CSV path with diagnosis/target label')
    ap.add_argument('--outdir', default='results/external/medical_screening')
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--label-column', default=None)
    ap.add_argument('--positive-label', default=None)
    ap.add_argument('--contamination', type=float, default=None)
    ap.add_argument('--include-slow', action='store_true')
    args = ap.parse_args()

    df = read_csv(args.data, args.max_rows)
    label_col = args.label_column or find_label_column(df, ['diagnosis', 'Diagnosis', 'target', 'Target', 'Class', 'label', 'outcome'])
    if label_col is None:
        raise SystemExit('Could not find a diagnosis/target label. Pass --label-column explicitly.')
    y = to_binary(df[label_col], args.positive_label)
    contamination = args.contamination or max(float(y.mean()), 1.0 / max(len(y), 1))
    X, cols = numeric_matrix(df, label_col=label_col, drop_cols=['id', 'ID'])
    X = scale_matrix(X)

    rows, scores = evaluate_external_baselines(X, y, contamination=contamination, include_slow=args.include_slow, robust='auto')
    out = Path(args.outdir)
    save_table(rows, out / 'metrics.csv')
    plot_metric(rows, 'roc_auc', out / 'roc_auc.png', 'Medical screening: ROC AUC')
    plot_metric(rows, 'pr_auc', out / 'pr_auc.png', 'Medical screening: PR AUC')
    plot_score_profile(scores, y, out / 'robust_score_profile.png', 'Medical robust score profile')
    write_markdown_summary(out / 'summary.md', 'Medical tabular screening', rows, [
        f'Rows: {len(df)}; features after preprocessing: {len(cols)}.',
        f'Positive rate/used contamination: {contamination:.5f}.',
        'Use this only as screening/diagnostics; clinical decisions require domain validation.',
    ])
    print_table(rows, 'medical screening benchmark')
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
