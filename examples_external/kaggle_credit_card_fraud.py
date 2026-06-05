#!/usr/bin/env python
"""Optional Kaggle example: Credit Card Fraud Detection.

Download the dataset manually from Kaggle, then run:

  python examples_external/kaggle_credit_card_fraud.py --data /path/to/creditcard.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
from _kaggle_utils import read_csv, find_label_column, numeric_matrix, scale_matrix, evaluate_external_baselines, save_table, print_table, plot_metric, plot_score_profile, write_markdown_summary


def main():
    ap = argparse.ArgumentParser(description='Kaggle Credit Card Fraud with robustcov')
    ap.add_argument('--data', required=True, help='Path to creditcard.csv with Class column')
    ap.add_argument('--outdir', default='results/external/credit_card_fraud')
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--contamination', type=float, default=None, help='Defaults to observed fraud rate')
    ap.add_argument('--include-slow', action='store_true', help='Include OneClassSVM')
    args = ap.parse_args()

    df = read_csv(args.data, args.max_rows)
    label_col = find_label_column(df, ['Class', 'isFraud', 'fraud', 'target'])
    if label_col is None:
        raise SystemExit('Could not find label column. Expected one of: Class, isFraud, fraud, target')
    y = df[label_col].astype(int).to_numpy()
    contamination = args.contamination or max(float(y.mean()), 1.0 / max(len(y), 1))
    X, cols = numeric_matrix(df, label_col=label_col)
    X = scale_matrix(X)

    rows, scores = evaluate_external_baselines(X, y, contamination=contamination, include_slow=args.include_slow, robust='fastmcd')
    out = Path(args.outdir)
    save_table(rows, out / 'metrics.csv')
    plot_metric(rows, 'pr_auc', out / 'pr_auc.png', 'Credit-card fraud: PR AUC')
    plot_metric(rows, 'f1', out / 'f1.png', 'Credit-card fraud: F1 at contamination threshold')
    plot_score_profile(scores, y, out / 'robust_score_profile.png', 'Credit-card fraud robust score profile')
    write_markdown_summary(out / 'summary.md', 'Credit-card fraud screening', rows, [
        f'Rows: {len(df)}; features after preprocessing: {len(cols)}.',
        f'Observed/used contamination: {contamination:.5f}.',
        'Use PR AUC as the main metric because fraud is extremely imbalanced.',
    ])
    print_table(rows, 'credit-card fraud benchmark')
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
