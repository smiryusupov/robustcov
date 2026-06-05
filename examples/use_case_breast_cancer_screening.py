"""Real ML use case: breast-cancer screening as anomaly detection.

This example uses sklearn's built-in breast-cancer dataset. It treats malignant
cases as the anomaly class and compares robust covariance scores with common
sklearn anomaly detectors.

Run:
    python examples/use_case_breast_cancer_screening.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import robustcov as rc
from _ml_use_case_utils import baseline_rows, print_rows, save_rows_csv, plot_metric_bars, plot_score_profile

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data = load_breast_cancer()
    # sklearn target: 0=malignant, 1=benign. Treat malignant as anomaly.
    labels = (data.target == 0).astype(int)
    X = StandardScaler().fit_transform(data.data)
    X = PCA(n_components=12, random_state=0).fit_transform(X)
    contamination = float(labels.mean())

    rows, est = baseline_rows(X, labels, contamination, robust_estimator='fastmcd')
    outdir = Path('results/use_cases/breast_cancer')
    outdir.mkdir(parents=True, exist_ok=True)
    print_rows('breast-cancer anomaly screening', rows)
    print(f'n={X.shape[0]}, p={X.shape[1]}, anomaly_fraction={contamination:.3f}')
    print(f'robust_radial_kurtosis={getattr(est, "radial_kurtosis_", np.nan):.3f}')

    save_rows_csv(rows, outdir / 'metrics.csv')
    plot_metric_bars(rows, 'f1', outdir / 'baseline_f1.png', title='Breast-cancer anomaly screening: F1')
    if hasattr(est, 'distances_'):
        rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / 'distance_panel.png', show=False)
        plot_score_profile(est.distances_, labels, outdir / 'score_profile.png', title='Breast-cancer robust-distance profile')
    print('saved diagnostics to', outdir)
