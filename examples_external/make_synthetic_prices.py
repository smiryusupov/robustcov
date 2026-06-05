#!/usr/bin/env python
"""Generate a small synthetic multi-asset price CSV for finance demos.

This lets users run the finance external examples without downloading market
or Kaggle data first.  The generated data contains correlated heavy-tailed
returns and a few injected stress windows so the robust-distance examples have
visible structure.

Example:
  python examples_external/make_synthetic_prices.py \
    --out examples_external/data/prices.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Generate synthetic multi-asset prices for robustcov finance demos')
    ap.add_argument('--out', default='examples_external/data/prices.csv')
    ap.add_argument('--n-days', type=int, default=900)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--assets', nargs='*', default=['SPY','QQQ','IWM','TLT','GLD','EFA','EEM','HYG'])
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f'pandas is required: {exc}')

    rng = np.random.default_rng(args.seed)
    p = len(args.assets)
    n = args.n_days
    if n < 120:
        raise SystemExit('--n-days should be at least 120 for meaningful stress windows')

    base_corr = 0.35 * np.ones((p, p)) + 0.65 * np.eye(p)
    L = np.linalg.cholesky(base_corr)
    returns = 0.00025 + 0.0095 * (rng.standard_t(df=4, size=(n, p)) @ L.T)

    # Deterministic stress regimes.  Use fractions so the script works for
    # shorter/longer series too.
    stress_starts = sorted({int(0.22*n), int(0.47*n), int(0.78*n)})
    for start in stress_starts:
        length = max(8, min(14, n - start - 1))
        if length <= 0:
            continue
        shock = rng.normal(-0.024, 0.018, size=(length, p))
        returns[start:start+length] += shock
        returns[start:start+length, :max(2, p//2)] *= 1.7

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range(args.start, periods=n, freq='B')
    df = pd.DataFrame(prices, columns=args.assets)
    df.insert(0, 'date', dates)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {out}')
    print(f'rows={len(df)}, assets={p}, start={df["date"].iloc[0].date()}, end={df["date"].iloc[-1].date()}')
    print('stress windows injected near rows:', ', '.join(map(str, stress_starts)))


if __name__ == '__main__':
    main()
