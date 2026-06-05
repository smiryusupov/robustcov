#!/usr/bin/env python
"""Optional external example: predictive-maintenance / sensor fault screening."""
from __future__ import annotations
import argparse
from pathlib import Path
from _kaggle_utils import read_csv, find_label_column, numeric_matrix, scale_matrix, evaluate_external_baselines, save_table, print_table, plot_metric, plot_score_profile, write_markdown_summary


def main():
    ap = argparse.ArgumentParser(description='Predictive maintenance anomaly screening with robustcov')
    ap.add_argument('--data', required=True, help='CSV path with a failure/target label')
    ap.add_argument('--outdir', default='results/external/predictive_maintenance')
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--label-column', default=None)
    ap.add_argument('--contamination', type=float, default=None)
    ap.add_argument('--include-slow', action='store_true')
    args = ap.parse_args()

    df = read_csv(args.data, args.max_rows)
    label_col = args.label_column or find_label_column(df, ['Machine failure', 'machine_failure', 'failure', 'Failure', 'Target', 'target', 'label'])
    if label_col is None:
        raise SystemExit('Could not find a failure label. Pass --label-column explicitly.')
    y = df[label_col].astype(int).to_numpy()
    contamination = args.contamination or max(float(y.mean()), 1.0 / max(len(y), 1))
    X, cols = numeric_matrix(df, label_col=label_col, drop_cols=['UDI', 'Product ID'])
    X = scale_matrix(X)

    rows, scores = evaluate_external_baselines(X, y, contamination=contamination, include_slow=args.include_slow, robust='auto')
    out = Path(args.outdir)
    save_table(rows, out / 'metrics.csv')
    plot_metric(rows, 'pr_auc', out / 'pr_auc.png', 'Predictive maintenance: PR AUC')
    plot_metric(rows, 'f1', out / 'f1.png', 'Predictive maintenance: F1 at threshold')
    plot_score_profile(scores, y, out / 'robust_score_profile.png', 'Equipment robust score profile')
    write_markdown_summary(out / 'summary.md', 'Predictive-maintenance fault screening', rows, [
        f'Rows: {len(df)}; features after preprocessing: {len(cols)}.',
        f'Failure rate/used contamination: {contamination:.5f}.',
        'A strong production workflow uses robustcov scores as early-warning features combined with supervised models.',
    ])
    print_table(rows, 'predictive maintenance benchmark')
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
