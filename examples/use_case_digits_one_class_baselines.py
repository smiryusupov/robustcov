"""Real ML use case: one-class digit anomaly detection.

The example keeps two digit classes from sklearn digits. One digit is the normal
class and the other is treated as an anomaly class. It compares robust distances
with standard sklearn anomaly baselines.

Run:
    python examples/use_case_digits_one_class_baselines.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import robustcov as rc
from _ml_use_case_utils import baseline_rows, print_rows, save_rows_csv, plot_metric_bars, plot_score_profile

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    normal_digit, anomaly_digit = 0, 1
    anomaly_count = 30
    rng = np.random.default_rng(0)
    digits = load_digits()
    normal_idx = np.where(digits.target == normal_digit)[0]
    anomaly_idx = np.where(digits.target == anomaly_digit)[0]
    anomaly_idx = rng.choice(anomaly_idx, size=anomaly_count, replace=False)
    idx = np.r_[normal_idx, anomaly_idx]
    labels = np.r_[np.zeros(normal_idx.size, dtype=int), np.ones(anomaly_idx.size, dtype=int)]
    X = StandardScaler().fit_transform(digits.data[idx])
    X = PCA(n_components=12, random_state=0).fit_transform(X)
    contamination = float(labels.mean())

    rows, est = baseline_rows(X, labels, contamination, robust_estimator='fastmcd')
    outdir = Path('results/use_cases/digits_one_class')
    outdir.mkdir(parents=True, exist_ok=True)
    print_rows('digits one-class anomaly detection', rows)
    print(f'normal_digit={normal_digit}, anomaly_digit={anomaly_digit}, anomaly_count={anomaly_count}, n={X.shape[0]}, p={X.shape[1]}')
    print(f'anomaly_fraction={contamination:.3f}, robust_radial_kurtosis={getattr(est, "radial_kurtosis_", np.nan):.3f}')

    save_rows_csv(rows, outdir / 'metrics.csv')
    plot_metric_bars(rows, 'f1', outdir / 'baseline_f1.png', title='Digits one-class anomaly detection: F1')
    if hasattr(est, 'distances_'):
        rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / 'distance_panel.png', show=False)
        plot_score_profile(est.distances_, labels, outdir / 'score_profile.png', title='Digits robust-distance profile')
    print('saved diagnostics to', outdir)
