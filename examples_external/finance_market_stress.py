#!/usr/bin/env python
"""Optional external example: finance market-stress anomaly detection.

Input can be either prices or returns in a CSV with a date column and one column
per asset.  The script fits a robust scatter estimator to multivariate returns
and ranks days by robust distance.

Example:

  python examples_external/finance_market_stress.py \
    --prices prices.csv \
    --outdir results/external/finance_market_stress
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from _kaggle_utils import ensure_deps


def _load_returns(path, date_col=None, input_kind='prices'):
    ensure_deps()
    import pandas as pd
    df = pd.read_csv(path)
    if date_col is None:
        date_col = next((c for c in df.columns if c.lower() in {'date','datetime','time','timestamp'}), None)
    dates = pd.to_datetime(df[date_col]) if date_col and date_col in df.columns else pd.RangeIndex(len(df))
    data = df.drop(columns=[date_col], errors='ignore').select_dtypes(include=['number']).copy()
    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if input_kind == 'prices':
        rets = np.log(data).diff().replace([np.inf, -np.inf], np.nan).dropna()
        dates = dates.iloc[rets.index] if hasattr(dates, 'iloc') else dates[rets.index]
    else:
        rets = data.dropna()
        dates = dates.iloc[rets.index] if hasattr(dates, 'iloc') else dates[rets.index]
    rets = rets.fillna(0.0)
    return rets, dates


def main():
    ap = argparse.ArgumentParser(description='Finance market-stress anomaly detection with robustcov')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prices', help='CSV with price levels, one column per asset')
    group.add_argument('--returns', help='CSV with returns, one column per asset')
    ap.add_argument('--date-column', default=None)
    ap.add_argument('--outdir', default='results/external/finance_market_stress')
    ap.add_argument('--alpha', type=float, default=0.975)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--estimator', choices=['cauchy','student','tyler','auto'], default='cauchy')
    args = ap.parse_args()

    ensure_deps()
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import robustcov as rc

    path = args.prices or args.returns
    input_kind = 'prices' if args.prices else 'returns'
    rets, dates = _load_returns(path, args.date_column, input_kind=input_kind)
    X = StandardScaler().fit_transform(rets.to_numpy(dtype=float))

    if args.estimator == 'auto':
        auto = rc.AutoRobustScatter(selection='diagnostic', random_state=0).fit(X)
        est = auto.estimator_
        est_name = f'AutoRobustScatter({auto.best_estimator_name_})'
    elif args.estimator == 'student':
        est = rc.StudentTScatter(df=3, alpha=0.05, warn_on_nonconvergence=False).fit(X)
        est_name = 'StudentTScatter'
    elif args.estimator == 'tyler':
        est = rc.RegularizedTyler(alpha=0.10).fit(X)
        est_name = 'RegularizedTyler'
    else:
        est = rc.RegularizedCauchy(alpha=0.10, warn_on_nonconvergence=False).fit(X)
        est_name = 'RegularizedCauchy'

    det = rc.RobustOutlierDetector(estimator=est, threshold='empirical', alpha=args.alpha).fit(X)
    scores = np.asarray(det.distances_, dtype=float)
    order = np.argsort(scores)[::-1]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    top = pd.DataFrame({
        'rank': np.arange(1, min(args.top_k, len(order)) + 1),
        'date': [str(pd.Timestamp(dates.iloc[i]).date()) if hasattr(dates, 'iloc') else str(dates[i]) for i in order[:args.top_k]],
        'robust_distance': scores[order[:args.top_k]],
    })
    top.to_csv(out / 'stress_days.csv', index=False)

    metrics = pd.DataFrame([{
        'method': est_name,
        'n_days': len(rets),
        'n_assets': rets.shape[1],
        'alpha': args.alpha,
        'detected_days': int((det.labels_ == -1).sum()),
        'threshold': float(det.threshold_),
        'max_distance': float(np.max(scores)),
        'median_distance': float(np.median(scores)),
        'radial_kurtosis': float(getattr(est, 'radial_kurtosis_', np.nan)),
        'condition_number': float(getattr(est, 'condition_number_', np.linalg.cond(est.covariance_ if hasattr(est,'covariance_') else est.shape_))),
    }])
    metrics.to_csv(out / 'metrics.csv', index=False)

    fig = plt.figure(figsize=(10, 4.8)); ax = fig.add_subplot(111)
    ax.plot(pd.to_datetime(dates), scores, linewidth=1.1)
    ax.axhline(det.threshold_, linestyle='--', linewidth=1.0, label=f'empirical {args.alpha:.3f} threshold')
    idx = np.where(det.labels_ == -1)[0]
    if idx.size:
        ax.scatter(pd.to_datetime(dates.iloc[idx] if hasattr(dates,'iloc') else np.asarray(dates)[idx]), scores[idx], s=18, marker='o', facecolors='none', edgecolors='black', label='flagged days')
    ax.set_title('Finance market stress: robust distance through time')
    ax.set_xlabel('date'); ax.set_ylabel('robust distance'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'robust_distance_timeseries.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
    show = top.iloc[::-1]
    ax.barh(show['date'], show['robust_distance'])
    ax.set_xlabel('robust distance'); ax.set_title('Top market-stress days')
    fig.tight_layout(); fig.savefig(out / 'top_stress_days.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
    cov = est.covariance_ if hasattr(est, 'covariance_') else est.shape_
    im = ax.imshow(cov, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{est_name} scatter/covariance heatmap')
    fig.tight_layout(); fig.savefig(out / 'covariance_heatmap.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    lines = [
        '# Finance market-stress anomaly detection', '',
        f'Estimator: **{est_name}**.',
        f'Days: {len(rets)}; assets: {rets.shape[1]}; alpha: {args.alpha:.3f}.',
        f'Flagged days: {int((det.labels_ == -1).sum())}.',
        '', '## Top stress days', '', top.to_markdown(index=False), '',
        'Interpretation: high robust distances indicate days whose cross-asset return vector is unusual relative to the robust central market regime.',
    ]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')

    print('finance market-stress anomaly detection')
    print(metrics.to_csv(index=False).strip())
    print('top stress days')
    print(top.head(10).to_csv(index=False).strip())
    print(f'saved outputs to {out}')


if __name__ == '__main__':
    main()
