#!/usr/bin/env python
"""Collect optional external/Kaggle results into a compact registry.

Usage:
  python examples_external/collect_external_results.py \
    --root results/external \
    --outdir results/external_registry
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kaggle_utils import ensure_deps, write_external_registry


def infer_task(name):
    n = name.lower()
    if 'fraud' in n: return 'fraud/anomaly detection'
    if 'network' in n or 'intrusion' in n: return 'network intrusion detection'
    if 'maintenance' in n: return 'predictive maintenance'
    if 'medical' in n: return 'medical screening'
    if 'finance' in n or 'portfolio' in n: return 'finance time-series anomaly detection'
    return 'external anomaly detection'


def main():
    ap = argparse.ArgumentParser(description='Collect robustcov external-result folders')
    ap.add_argument('--root', default='results/external')
    ap.add_argument('--outdir', default='results/external_registry')
    args = ap.parse_args()
    ensure_deps()
    import pandas as pd
    rows = []
    root = Path(args.root)
    for metrics_path in sorted(root.glob('*/metrics.csv')):
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        # If supervised metrics exist, choose by PR AUC/F1. Otherwise keep the first method row.
        sort_cols = [c for c in ['pr_auc', 'f1'] if c in df.columns]
        if sort_cols:
            best = df.sort_values(sort_cols, ascending=False).iloc[0]
            pr_auc = float(best.get('pr_auc', float('nan')))
            f1 = float(best.get('f1', float('nan')))
        else:
            best = df.iloc[0]
            pr_auc = float('nan'); f1 = float('nan')
        rows.append({
            'dataset': metrics_path.parent.name.replace('_', ' '),
            'task': infer_task(metrics_path.parent.name),
            'best_method': str(best.get('method', best.get('estimator', ''))),
            'best_pr_auc': pr_auc,
            'best_f1': f1,
            'best_seconds': float(best.get('seconds', float('nan'))),
            'path': str(metrics_path.parent),
        })
    write_external_registry(rows, args.outdir)
    print(f'collected {len(rows)} external result folders into {args.outdir}')


if __name__ == '__main__':
    main()
