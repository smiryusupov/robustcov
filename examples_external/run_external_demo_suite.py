#!/usr/bin/env python
"""Run a small optional external-demo suite.

The default ``--synthetic`` mode is fully reproducible and does not require
Kaggle downloads.  It generates synthetic prices, runs the two finance examples,
and collects the external registry.  Kaggle datasets should still be run
one-by-one because they require manual downloads and licenses.

Example:
  python examples_external/run_external_demo_suite.py --synthetic
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def _run(cmd):
    print('$ ' + ' '.join(map(str, cmd)))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    ap = argparse.ArgumentParser(description='Run robustcov optional external demo suite')
    ap.add_argument('--synthetic', action='store_true', help='Run reproducible synthetic finance external examples')
    ap.add_argument('--prices', default='examples_external/data/prices.csv', help='Price CSV to use/generate in synthetic mode')
    ap.add_argument('--outroot', default='results/external')
    ap.add_argument('--registry-outdir', default='results/external_registry')
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--step', type=int, default=5)
    args = ap.parse_args()

    if not args.synthetic:
        raise SystemExit('Only --synthetic mode is automated. Kaggle datasets require manual downloads; run their scripts individually.')

    py = sys.executable
    prices = Path(args.prices)
    outroot = Path(args.outroot)

    _run([py, 'examples_external/make_synthetic_prices.py', '--out', str(prices)])
    _run([py, 'examples_external/finance_market_stress.py', '--prices', str(prices), '--outdir', str(outroot / 'finance_market_stress')])
    _run([py, 'examples_external/finance_rolling_window_anomaly.py', '--prices', str(prices), '--window', str(args.window), '--step', str(args.step), '--outdir', str(outroot / 'finance_rolling_window')])
    _run([py, 'examples_external/collect_external_results.py', '--root', str(outroot), '--outdir', args.registry_outdir])
    print(f'\nexternal synthetic demo suite complete')
    print(f'prices: {prices}')
    print(f'results: {outroot}')
    print(f'registry: {args.registry_outdir}')


if __name__ == '__main__':
    main()
