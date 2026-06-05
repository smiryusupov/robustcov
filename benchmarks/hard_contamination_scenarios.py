"""Harder synthetic contamination scenarios for robust covariance estimators.

Run:
    python benchmarks/hard_contamination_scenarios.py --csv results/hard_scenarios.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import robustcov as rc


def rel_fro(cov, truth):
    return np.linalg.norm(cov - truth, ord='fro') / np.linalg.norm(truth, ord='fro')


def base_data(n, p, seed):
    rng = np.random.default_rng(seed)
    Sigma = 0.65 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X_clean = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    return X_clean, Sigma, rng


def contaminate(X_clean, scenario, frac, rng):
    X = X_clean.copy()
    n, p = X.shape
    m = int(round(frac * n))
    is_out = np.zeros(n, dtype=bool)
    if m == 0:
        return X, is_out
    idx = rng.choice(n, size=m, replace=False)
    is_out[idx] = True
    if scenario == 'mean_shift':
        X[idx] += rng.normal(8.0, 1.5, size=(m, p))
    elif scenario == 'clustered':
        direction = np.zeros(p)
        direction[: min(3, p)] = 1.0
        direction = direction / np.linalg.norm(direction)
        center = 8.0 * direction
        X[idx] = rng.multivariate_normal(center, 0.15 * np.eye(p), size=m)
    elif scenario == 'variance':
        X[idx] *= rng.uniform(5.0, 9.0, size=(m, 1))
    elif scenario == 'leverage':
        direction = np.zeros(p)
        direction[0] = 1.0
        X[idx] = rng.normal(0.0, 0.2, size=(m, p))
        X[idx] += rng.normal(12.0, 2.0, size=(m, 1)) * direction
    elif scenario == 'heavy_tail_inliers':
        # Not label contamination: all observations become heavy-tailed elliptical.
        scales = np.sqrt(rng.chisquare(df=3, size=n) / 3.0)
        X = X_clean / np.maximum(scales[:, None], 1e-8)
        is_out[:] = False
    else:
        raise ValueError(f'unknown scenario {scenario}')
    return X, is_out


def maybe_sklearn():
    out = {}
    try:
        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        out['sklearn Empirical'] = lambda: EmpiricalCovariance()
        out['sklearn MinCovDet'] = lambda: MinCovDet(random_state=0)
    except Exception:
        pass
    return out


def support_purity(est, is_out):
    supp = getattr(est, 'support_', None)
    if supp is None or np.sum(supp) == 0 or np.sum(is_out) == 0:
        return '', '', ''
    supp = np.asarray(supp, dtype=bool)
    kept_out = np.sum(supp & is_out)
    purity = np.sum(supp & ~is_out) / np.sum(supp)
    leakage = kept_out / np.sum(is_out)
    return f'{purity:.4f}', f'{leakage:.4f}', int(np.sum(supp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--contamination', type=float, default=0.20)
    parser.add_argument('--quality', choices=['fast', 'balanced', 'high'], default='balanced')
    parser.add_argument('--csv', type=str, default='')
    args = parser.parse_args()

    scenarios = ['mean_shift', 'clustered', 'variance', 'leverage', 'heavy_tail_inliers']
    methods = {
        'robustcov FastMCD': lambda: rc.FastMCD(quality=args.quality, random_state=0),
        'robustcov RegTyler': lambda: rc.RegularizedTyler(alpha=0.05, scale_correction='radial_median'),
    }
    methods.update(maybe_sklearn())

    rows = []
    for scenario in scenarios:
        X_clean, Sigma, rng = base_data(args.n, args.p, seed=100)
        X, is_out = contaminate(X_clean, scenario, args.contamination, rng)
        for name, factory in methods.items():
            try:
                t0 = time.perf_counter()
                est = factory().fit(X)
                seconds = time.perf_counter() - t0
                purity, leakage, support_size = support_purity(est, is_out)
                rows.append({
                    'scenario': scenario,
                    'contamination': args.contamination,
                    'method': name,
                    'rel_fro_error': f'{rel_fro(est.covariance_, Sigma):.4f}',
                    'seconds': f'{seconds:.6f}',
                    'support_purity': purity,
                    'outlier_leakage': leakage,
                    'support_size': support_size,
                    'radial_kurtosis': f'{getattr(est, "radial_kurtosis_", np.nan):.4f}' if hasattr(est, 'radial_kurtosis_') else '',
                })
            except Exception as exc:
                rows.append({
                    'scenario': scenario,
                    'contamination': args.contamination,
                    'method': name,
                    'rel_fro_error': 'FAILED',
                    'seconds': f'{type(exc).__name__}: {exc}',
                    'support_purity': '',
                    'outlier_leakage': '',
                    'support_size': '',
                    'radial_kurtosis': '',
                })

    fieldnames = ['scenario', 'contamination', 'method', 'rel_fro_error', 'seconds', 'support_purity', 'outlier_leakage', 'support_size', 'radial_kurtosis']
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
