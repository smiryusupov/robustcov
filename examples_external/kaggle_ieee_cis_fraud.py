#!/usr/bin/env python
"""Optional external example: IEEE-CIS fraud-style transaction screening.

Run against Kaggle's train_transaction.csv. Optional identity features can be
merged later, but this script keeps the first version simple and reproducible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from _kaggle_utils import read_csv, find_label_column, numeric_matrix, scale_matrix, evaluate_external_baselines, save_table, print_table, plot_metric, plot_score_profile, write_markdown_summary


def main():
    ap = argparse.ArgumentParser(description='IEEE-CIS transaction fraud with robustcov')
    ap.add_argument('--data', required=True, help='Path to train_transaction.csv')
    ap.add_argument('--outdir', default='results/external/ieee_cis_fraud')
    ap.add_argument('--max-rows', type=int, default=100000)
    ap.add_argument('--contamination', type=float, default=None)
    ap.add_argument('--include-slow', action='store_true')
    args = ap.parse_args()

    df = read_csv(args.data, args.max_rows)
    label_col = find_label_column(df, ['isFraud', 'Class', 'fraud', 'target'])
    if label_col is None:
        raise SystemExit('Could not find label column. Expected isFraud or target-like column.')
    y = df[label_col].astype(int).to_numpy()
    contamination = args.contamination or max(float(y.mean()), 1.0 / max(len(y), 1))
    X, cols = numeric_matrix(df, label_col=label_col, drop_cols=['TransactionID'])
    X = scale_matrix(X)

    rows, scores = evaluate_external_baselines(X, y, contamination=contamination, include_slow=args.include_slow, robust='cauchy')
    out = Path(args.outdir)
    save_table(rows, out / 'metrics.csv')
    plot_metric(rows, 'pr_auc', out / 'pr_auc.png', 'IEEE-CIS fraud: PR AUC')
    plot_metric(rows, 'f1', out / 'f1.png', 'IEEE-CIS fraud: F1 at threshold')
    plot_score_profile(scores, y, out / 'robust_score_profile.png', 'IEEE-CIS robust transaction score profile')
    write_markdown_summary(out / 'summary.md', 'IEEE-CIS fraud screening', rows, [
        f'Rows: {len(df)}; features after preprocessing: {len(cols)}.',
        f'Observed/used contamination: {contamination:.5f}.',
        'This example treats robustcov scores as an interpretable anomaly-screening feature, not a full competition-winning supervised model.',
    ])
    print_table(rows, 'IEEE-CIS fraud benchmark')
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
