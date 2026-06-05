#!/usr/bin/env python
"""Optional external example: rolling-window finance anomaly detection.

The script converts a multivariate return series into rolling-window features
(volatility, mean return, max absolute return, and average correlation), then
uses robust distances to flag unusual market regimes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from finance_market_stress import _load_returns
from _kaggle_utils import ensure_deps


def _window_features(rets, window, step):
    rows, labels = [], []
    values = rets.to_numpy(dtype=float)
    dates = rets.index
    for start in range(0, len(rets) - window + 1, step):
        block = values[start:start+window]
        mean = block.mean(axis=0)
        vol = block.std(axis=0)
        max_abs = np.max(np.abs(block), axis=0)
        corr = np.corrcoef(block, rowvar=False)
        if corr.ndim == 2:
            iu = np.triu_indices_from(corr, k=1)
            avg_abs_corr = np.nanmean(np.abs(corr[iu])) if iu[0].size else 0.0
        else:
            avg_abs_corr = 0.0
        feat = np.concatenate([mean, vol, max_abs, [avg_abs_corr]])
        rows.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))
        labels.append((start, start+window-1))
    return np.asarray(rows), labels


def main():
    ap = argparse.ArgumentParser(description='Rolling-window finance anomaly detection with robustcov')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prices', help='CSV with price levels')
    group.add_argument('--returns', help='CSV with returns')
    ap.add_argument('--date-column', default=None)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.975)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--outdir', default='results/external/finance_rolling_window')
    args = ap.parse_args()

    ensure_deps()
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import robustcov as rc

    path = args.prices or args.returns
    input_kind = 'prices' if args.prices else 'returns'
    rets, dates = _load_returns(path, args.date_column, input_kind=input_kind)
    rets = rets.reset_index(drop=True)
    Xraw, windows = _window_features(rets, args.window, args.step)
    if len(Xraw) < 5:
        raise SystemExit('Not enough windows; reduce --window or --step.')
    X = StandardScaler().fit_transform(Xraw)

    est = rc.AutoRobustScatter(selection='diagnostic', random_state=0).fit(X).estimator_
    det = rc.RobustOutlierDetector(estimator=est, threshold='empirical', alpha=args.alpha).fit(X)
    scores = np.asarray(det.distances_, dtype=float)
    order = np.argsort(scores)[::-1]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, wi in enumerate(order[:args.top_k], start=1):
        s, e = windows[wi]
        start_date = str(pd.Timestamp(dates.iloc[s]).date()) if hasattr(dates, 'iloc') else str(s)
        end_date = str(pd.Timestamp(dates.iloc[e]).date()) if hasattr(dates, 'iloc') else str(e)
        rows.append({'rank': rank, 'start_date': start_date, 'end_date': end_date, 'robust_distance': scores[wi]})
    top = pd.DataFrame(rows); top.to_csv(out / 'stress_windows.csv', index=False)
    metrics = pd.DataFrame([{
        'method': type(est).__name__, 'n_windows': len(X), 'window': args.window, 'step': args.step,
        'n_assets': rets.shape[1], 'detected_windows': int((det.labels_ == -1).sum()),
        'threshold': float(det.threshold_), 'max_distance': float(scores.max()),
        'radial_kurtosis': float(getattr(est, 'radial_kurtosis_', np.nan)),
    }])
    metrics.to_csv(out / 'metrics.csv', index=False)

    xdates = [pd.Timestamp(dates.iloc[e]) if hasattr(dates, 'iloc') else e for _, e in windows]
    fig = plt.figure(figsize=(10,4.8)); ax = fig.add_subplot(111)
    ax.plot(xdates, scores, linewidth=1.1)
    ax.axhline(det.threshold_, linestyle='--', linewidth=1.0, label=f'empirical {args.alpha:.3f} threshold')
    idx = np.where(det.labels_ == -1)[0]
    if idx.size:
        ax.scatter([xdates[i] for i in idx], scores[idx], s=18, facecolors='none', edgecolors='black', label='flagged windows')
    ax.set_title('Rolling-window market-regime anomaly score')
    ax.set_xlabel('window end date'); ax.set_ylabel('robust distance'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'rolling_distance_timeseries.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    fig = plt.figure(figsize=(8,5)); ax = fig.add_subplot(111)
    show = top.iloc[::-1].copy(); show['window'] = show['start_date'] + '→' + show['end_date']
    ax.barh(show['window'], show['robust_distance'])
    ax.set_xlabel('robust distance'); ax.set_title('Top anomalous rolling windows')
    fig.tight_layout(); fig.savefig(out / 'top_stress_windows.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    (out / 'summary.md').write_text('\n'.join([
        '# Rolling-window finance anomaly detection', '',
        f'Estimator: **{type(est).__name__}**.',
        f'Windows: {len(X)}; assets: {rets.shape[1]}; window length: {args.window}; step: {args.step}.',
        '', '## Top anomalous windows', '', top.to_markdown(index=False), '',
        'Interpretation: high-scoring windows are periods whose volatility/correlation structure is unusual relative to the robust central market regime.',
    ]) + '\n')

    print('rolling-window finance anomaly detection')
    print(metrics.to_csv(index=False).strip())
    print('top anomalous windows')
    print(top.head(10).to_csv(index=False).strip())
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
