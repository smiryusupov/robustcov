"""Real ML use case: wine class screening.

This uses sklearn's wine dataset and treats one cultivar as the anomaly class.
The goal is to show robust-distance screening on a small tabular dataset with
correlated chemical measurements.

Run:
    python examples/use_case_wine_class_screening.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import robustcov as rc
from _ml_use_case_utils import baseline_rows, print_rows, save_rows_csv, plot_metric_bars, plot_score_profile

if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data = load_wine()
    normal_class, anomaly_class = 0, 1
    anomaly_count = 10
    rng = np.random.default_rng(0)
    normal_idx = np.where(data.target == normal_class)[0]
    anomaly_idx = np.where(data.target == anomaly_class)[0]
    anomaly_idx = rng.choice(anomaly_idx, size=anomaly_count, replace=False)
    idx = np.r_[normal_idx, anomaly_idx]
    labels = np.r_[np.zeros(normal_idx.size, dtype=int), np.ones(anomaly_idx.size, dtype=int)]
    X = StandardScaler().fit_transform(data.data[idx])
    X = PCA(n_components=8, random_state=0).fit_transform(X)
    contamination = float(labels.mean())

    rows, est = baseline_rows(X, labels, contamination, robust_estimator='auto')
    outdir = Path('results/use_cases/wine_class')
    outdir.mkdir(parents=True, exist_ok=True)
    print_rows('wine class screening', rows)
    print(f'normal_class={normal_class}, anomaly_class={anomaly_class}, anomaly_count={anomaly_count}, n={X.shape[0]}, p={X.shape[1]}, anomaly_fraction={contamination:.3f}')
    if hasattr(est, 'score_'):
        print(f'auto_detector_estimators={len(est.estimators_)}')
        plot_score_profile(est.score_, labels, outdir / 'score_profile.png', title='Wine class robust ensemble score')
    save_rows_csv(rows, outdir / 'metrics.csv')
    plot_metric_bars(rows, 'f1', outdir / 'baseline_f1.png', title='Wine class screening: F1')
    # For a distance panel, fit a single Cauchy scatter estimator for visual diagnostics.
    scatter = rc.RegularizedCauchy(alpha=0.10, warn_on_nonconvergence=False).fit(X)
    rc.plot_robust_distance_panel(scatter, labels=labels, output_path=outdir / 'distance_panel.png', show=False)
    print('saved diagnostics to', outdir)
