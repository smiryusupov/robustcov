"""Use case: robust input geometry for Gaussian-process kernels.

robustcov does not implement Gaussian-process regression, kernel ridge
regression, posterior inference, likelihoods, Bayesian optimization, or training
loops. It only estimates a robust full-matrix input metric that can be used by
existing GP/kernel libraries.

This example shows a failure mode: a small number of contaminated design points
inflate the classical empirical covariance in a direction that is important for
the response. A GP kernel using that non-robust metric oversmooths. The robust
metric is less warped by the contaminated input geometry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

import robustcov as rc
from robustcov.sklearn_kernels import RobustMahalanobisRBF


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_gp_with_metric(X_train, y_train, precision, center):
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RobustMahalanobisRBF(
            precision=precision,
            center=center,
            length_scale=1.0,
            length_scale_bounds="fixed",
        )
        + WhiteKernel(0.04, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    return gp.fit(X_train, y_train)


def _cov_ellipse(mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)


def main():
    rng = np.random.default_rng(42)

    # Clean design: response varies mainly in x1.
    n_clean = 90
    x0 = rng.normal(0.0, 1.0, size=n_clean)
    x1 = rng.normal(0.0, 0.18, size=n_clean)
    X_clean = np.column_stack([x0, x1])
    y_clean = np.sin(4.0 * x1) + 0.10 * rng.normal(size=n_clean)

    # Contaminated design points: far in x1 with unrelated responses.
    n_bad = 12
    X_bad = np.column_stack(
        [
            rng.normal(0.0, 1.0, size=n_bad),
            rng.normal(0.0, 4.0, size=n_bad),
        ]
    )
    y_bad = rng.normal(0.0, 0.2, size=n_bad)

    X_train = np.vstack([X_clean, X_bad])
    y_train = np.concatenate([y_clean, y_bad])

    # Test only on the intended clean regime.
    n_test = 300
    xt0 = rng.normal(0.0, 1.0, size=n_test)
    xt1 = rng.normal(0.0, 0.18, size=n_test)
    X_test = np.column_stack([xt0, xt1])
    y_test = np.sin(4.0 * xt1)

    # Non-robust input geometry: empirical covariance of all training inputs.
    empirical_center = X_train.mean(axis=0)
    empirical_cov = np.cov(X_train, rowvar=False) + 1e-8 * np.eye(X_train.shape[1])
    empirical_precision = np.linalg.pinv(empirical_cov)

    # Robust input geometry.
    metric = rc.RobustInputMetric(
        estimator=rc.FastMCD(contamination=0.12, quality="balanced", random_state=0),
    ).fit(X_train)

    gp_empirical = _fit_gp_with_metric(
        X_train, y_train, empirical_precision, empirical_center
    )
    gp_robust = _fit_gp_with_metric(
        X_train, y_train, metric.precision_, metric.location_
    )

    pred_empirical = gp_empirical.predict(X_test)
    pred_robust = gp_robust.predict(X_test)

    rmse_empirical = _rmse(y_test, pred_empirical)
    rmse_robust = _rmse(y_test, pred_robust)

    print("robust GP kernel input-metric example")
    print(f"gp_empirical_input_covariance_rmse={rmse_empirical:.4f}")
    print(f"gp_robust_input_covariance_rmse={rmse_robust:.4f}")
    print("empirical_covariance_diag=", np.round(np.diag(empirical_cov), 4))
    print("robust_covariance_diag=   ", np.round(np.diag(metric.covariance_), 4))

    # Prediction view only in clean region.
    x1_grid = np.linspace(-0.7, 0.7, 300)
    X_grid = np.column_stack([np.zeros_like(x1_grid), x1_grid])
    y_true_grid = np.sin(4.0 * x1_grid)

    mean_empirical, std_empirical = gp_empirical.predict(X_grid, return_std=True)
    mean_robust, std_robust = gp_robust.predict(X_grid, return_std=True)

    outdir = Path("results/use_cases/gp_robust_input_metric")
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))

    # Left panel: geometry
    ax1 = fig.add_subplot(121)
    ax1.scatter(X_clean[:, 0], X_clean[:, 1], s=22, alpha=0.6, label="clean training rows")
    ax1.scatter(X_bad[:, 0], X_bad[:, 1], s=50, marker="x", label="contaminated rows")
    ax1.add_patch(
        _cov_ellipse(empirical_center, empirical_cov, n_std=2.0, linestyle="--", linewidth=2, label="empirical covariance")
    )
    ax1.add_patch(
        _cov_ellipse(metric.location_, metric.covariance_, n_std=2.0, linestyle="-.", linewidth=2, label="robust covariance")
    )
    ax1.set_title("Input geometry")
    ax1.set_xlabel("x0")
    ax1.set_ylabel("x1")
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend(loc="upper left", fontsize=8)

    # Right panel: prediction in clean region
    ax2 = fig.add_subplot(122)
    ax2.plot(x1_grid, y_true_grid, label="clean signal")
    ax2.plot(x1_grid, mean_empirical, linestyle="--", label="GP with empirical input covariance")
    ax2.plot(x1_grid, mean_robust, linestyle="-.", label="GP with robust input covariance")
    ax2.fill_between(
        x1_grid,
        mean_empirical - 2.0 * std_empirical,
        mean_empirical + 2.0 * std_empirical,
        alpha=0.12,
    )
    ax2.fill_between(
        x1_grid,
        mean_robust - 2.0 * std_robust,
        mean_robust + 2.0 * std_robust,
        alpha=0.12,
    )
    ax2.scatter(X_clean[:, 1], y_clean, s=18, alpha=0.55, label="clean training rows")
    ax2.scatter(X_bad[:, 1], y_bad, s=28, marker="x", label="contaminated rows")
    ax2.set_title("Prediction in clean region")
    ax2.set_xlabel("important input direction x1")
    ax2.set_ylabel("response")
    ax2.set_xlim(-0.75, 0.75)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Robust input covariance can protect GP kernel geometry")
    fig.tight_layout()
    fig.savefig(outdir / "kernel_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved diagnostics to", outdir)


if __name__ == "__main__":
    main()