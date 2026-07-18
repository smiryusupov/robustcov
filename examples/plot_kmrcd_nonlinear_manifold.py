"""Kernel MRCD on a curved inlier manifold.

The regular observations follow a noisy parabola.  The injected outliers lie
inside the broad linear covariance envelope but away from the curve.  This is a
case where a nonlinear kernel can add information that ordinary Mahalanobis
geometry cannot represent.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import robustcov as rc


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_inliers = 240
    x = rng.uniform(-2.5, 2.5, n_inliers)
    inliers = np.column_stack([
        x,
        0.55 * x**2 + 0.08 * rng.normal(size=n_inliers),
    ])

    n_outliers = 50
    xo = rng.uniform(-1.8, 1.8, n_outliers)
    yo = rng.uniform(0.2, 2.0, n_outliers)
    close = np.abs(yo - 0.55 * xo**2) < 0.4
    while np.any(close):
        yo[close] = rng.uniform(0.2, 2.0, np.count_nonzero(close))
        close = np.abs(yo - 0.55 * xo**2) < 0.4
    outliers = np.column_stack([xo, yo])

    X = np.vstack([inliers, outliers])
    labels = np.r_[np.zeros(n_inliers, dtype=bool), np.ones(n_outliers, dtype=bool)]
    return X, labels


def main() -> None:
    X, labels = make_data()
    contamination = float(labels.mean())

    linear = rc.MRCD(
        contamination=contamination,
        n_init=32,
        n_best=5,
        initial_c_steps=2,
        max_iter=50,
        random_state=0,
    ).fit(X)

    kernel = rc.KMRCD(
        kernel="rbf",
        gamma=2.0,
        contamination=contamination,
        n_init=32,
        n_best=5,
        initial_c_steps=2,
        max_iter=50,
        random_state=0,
    ).fit(X)

    linear_auc = auc(labels, linear.distances_)
    kernel_auc = auc(labels, kernel.distances_)

    result_dir = Path("results/use_cases/kmrcd_nonlinear_manifold")
    result_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(-2.8, 2.8, 180)
    y_grid = np.linspace(-0.4, 3.8, 160)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    linear_grid = linear.mahalanobis(grid).reshape(xx.shape)
    kernel_grid = kernel.mahalanobis(grid).reshape(xx.shape)

    for name, values, title in [
        ("linear_distance_contours.png", linear_grid, "Linear MRCD distance"),
        ("kernel_distance_contours.png", kernel_grid, "RBF KMRCD distance"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        levels = np.quantile(values[np.isfinite(values)], [0.35, 0.55, 0.70, 0.82, 0.90, 0.96])
        ax.contour(xx, yy, values, levels=np.unique(levels), linewidths=1.0)
        ax.scatter(X[~labels, 0], X[~labels, 1], s=16, alpha=0.72, label="regular")
        ax.scatter(X[labels, 0], X[labels, 1], s=28, marker="x", label="injected outlier")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.set_title(title)
        ax.legend(loc="upper center", ncol=2)
        fig.tight_layout()
        fig.savefig(result_dir / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(2)
    ax.bar(positions, [linear_auc, kernel_auc])
    ax.set_xticks(positions, ["MRCD", "KMRCD (RBF)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Outlier AUROC")
    ax.set_title("Off-manifold outlier ranking")
    for index, value in enumerate([linear_auc, kernel_auc]):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(result_dir / "auc_comparison.png", dpi=150)
    plt.close(fig)

    gamma_grid = np.array([0.15, 0.3, 0.6, 1.0, 2.0, 4.0])
    gamma_auc = []
    for gamma in gamma_grid:
        fitted = rc.KMRCD(
            kernel="rbf",
            gamma=float(gamma),
            contamination=contamination,
            n_init=16,
            n_best=4,
            initial_c_steps=2,
            max_iter=40,
            random_state=0,
        ).fit(X)
        gamma_auc.append(auc(labels, fitted.distances_))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(gamma_grid, gamma_auc, marker="o")
    ax.axvline(kernel.gamma_, linestyle="--", linewidth=1.0, label="example setting")
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("RBF gamma")
    ax.set_ylabel("Outlier AUROC")
    ax.set_title("Bandwidth sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "bandwidth_sensitivity.png", dpi=150)
    plt.close(fig)

    metrics = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "outlier_fraction": contamination,
        "linear_mrcd_auc": linear_auc,
        "rbf_kmrcd_auc": kernel_auc,
        "rbf_gamma": kernel.gamma_,
        "regularization": kernel.regularization_,
        "outliers_in_support": int(kernel.support_[labels].sum()),
    }
    with (result_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    print(f"observations: {X.shape[0]}")
    print(f"injected outliers: {labels.sum()}")
    print(f"linear MRCD AUROC: {linear_auc:.3f}")
    print(f"RBF KMRCD AUROC: {kernel_auc:.3f}")
    print(f"RBF gamma: {kernel.gamma_:.3f}")
    print(f"feature-space regularization: {kernel.regularization_:.4f}")
    print(f"injected outliers retained in support: {kernel.support_[labels].sum()}")


if __name__ == "__main__":
    main()
