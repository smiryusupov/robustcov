#!/usr/bin/env python
"""Optional external example: network intrusion anomaly screening."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from _kaggle_utils import read_csv, find_label_column, numeric_matrix, scale_matrix, evaluate_external_baselines, save_table, print_table, plot_metric, plot_score_profile, write_markdown_summary


def labels_to_binary(series):
    s = series.astype(str).str.lower()
    normal_names = {'normal', 'benign', '0', 'false'}
    return (~s.isin(normal_names)).astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser(description='Network intrusion anomaly screening with robustcov')
    ap.add_argument('--data', required=True, help='CSV path, e.g. NSL-KDD/UNSW/CIC-style feature table')
    ap.add_argument('--outdir', default='results/external/network_intrusion')
    ap.add_argument('--max-rows', type=int, default=100000)
    ap.add_argument('--label-column', default=None)
    ap.add_argument('--contamination', type=float, default=None)
    ap.add_argument('--include-slow', action='store_true')
    args = ap.parse_args()

    df = read_csv(args.data, args.max_rows)
    label_col = args.label_column or find_label_column(df, ['Label', 'label', 'class', 'attack', 'target', 'outcome'])
    if label_col is None:
        raise SystemExit('Could not find a label column. Pass --label-column explicitly.')
    y = labels_to_binary(df[label_col])
    contamination = args.contamination or max(float(np.mean(y)), 1.0 / max(len(y), 1))
    X, cols = numeric_matrix(df, label_col=label_col)
    X = scale_matrix(X)

    rows, scores = evaluate_external_baselines(X, y, contamination=contamination, include_slow=args.include_slow, robust='fastmcd')
    out = Path(args.outdir)
    save_table(rows, out / 'metrics.csv')
    plot_metric(rows, 'roc_auc', out / 'roc_auc.png', 'Network intrusion: ROC AUC')
    plot_metric(rows, 'f1', out / 'f1.png', 'Network intrusion: F1 at threshold')
    plot_score_profile(scores, y, out / 'robust_score_profile.png', 'Network traffic robust score profile')
    write_markdown_summary(out / 'summary.md', 'Network intrusion screening', rows, [
        f'Rows: {len(df)}; features after preprocessing: {len(cols)}.',
        f'Positive class fraction/used contamination: {contamination:.5f}.',
        'Robust distances are most useful here as fast interpretable scores over traffic-summary features.',
    ])
    print_table(rows, 'network intrusion benchmark')
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
