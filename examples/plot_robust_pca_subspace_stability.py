# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the stability of contaminated yield-curve factors.

The example compares ordinary and robust scatter PCA on synthetic daily yield
changes. A small set of quote dislocations makes the empirical factor basis
sensitive to which days are resampled. RobustPCA is evaluated with the same
bootstrap rows and reports loading intervals together with principal angles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


class EmpiricalScatter:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = np.mean(X, axis=0)
        centered = X - self.location_
        self.covariance_ = centered.T @ centered / X.shape[0]
        return self


def _curve_basis(maturities: np.ndarray, tau: float = 2.5) -> np.ndarray:
    scaled = maturities / tau
    slope = (1.0 - np.exp(-scaled)) / scaled
    curvature = slope - np.exp(-scaled)
    raw = np.column_stack([np.ones_like(maturities), slope, curvature])
    basis, _ = np.linalg.qr(raw)
    return basis[:, :3]


def make_data(seed: int = 31):
    rng = np.random.default_rng(seed)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    basis = _curve_basis(maturities)
    n_days = 190
    factors = rng.standard_t(df=7, size=(n_days, 3)) * np.array([2.8, 1.6, 0.8])
    clean = factors @ basis.T + rng.normal(scale=0.14, size=(n_days, maturities.size))
    observed = clean.copy()

    quote_rows = rng.choice(n_days, size=28, replace=False)
    for row in quote_rows:
        affected = rng.choice(maturities.size, size=rng.integers(1, 4), replace=False)
        observed[row, affected] += rng.choice([-1.0, 1.0], size=affected.size) * rng.uniform(5.0, 9.0, size=affected.size)

    return maturities, observed, clean, basis, quote_rows


def _projection_error(components: np.ndarray, basis: np.ndarray) -> float:
    fitted = components.T @ components
    truth = basis @ basis.T
    return float(np.linalg.norm(fitted - truth, ord="fro"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/use_cases/robust_pca_subspace_stability")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    maturities, X, _, true_basis, quote_rows = make_data()
    empirical_pca = rc.RobustPCA(
        n_components=3,
        estimator=EmpiricalScatter(),
        store_scores=False,
    )
    robust_pca = rc.RobustPCA(
        n_components=3,
        estimator=rc.FastMCD(
            contamination=0.18,
            n_init=30,
            quality="fast",
            random_state=0,
            n_jobs=2,
        ),
        store_scores=False,
    )

    empirical = rc.SubspaceStability(
        pca=empirical_pca,
        n_resamples=60,
        confidence_level=0.90,
        alignment="procrustes",
        random_state=7,
        min_successful_resamples=50,
    ).fit(X)
    robust = rc.SubspaceStability(
        pca=robust_pca,
        n_resamples=60,
        confidence_level=0.90,
        alignment="procrustes",
        random_state=7,
        min_successful_resamples=50,
    ).fit(X)

    empirical_error = _projection_error(empirical.components_, true_basis)
    robust_error = _projection_error(robust.components_, true_basis)

    print("Bootstrap stability of yield-curve factors")
    print("==========================================")
    print(f"observations: {X.shape[0]}")
    print(f"maturities: {X.shape[1]}")
    print(f"quote-dislocation days: {quote_rows.size}")
    print(f"successful bootstrap fits, empirical / robust: {empirical.n_successful_resamples_} / {robust.n_successful_resamples_}")
    print(f"median largest principal angle, empirical / robust: {empirical.median_max_principal_angle_degrees_:.3f} / {robust.median_max_principal_angle_degrees_:.3f} degrees")
    print(f"90% upper largest angle, empirical / robust: {empirical.max_principal_angle_interval_degrees_[1]:.3f} / {robust.max_principal_angle_interval_degrees_[1]:.3f} degrees")
    print(f"reference subspace error, empirical / robust: {empirical_error:.3f} / {robust_error:.3f}")
    print(f"stable loading count, PC1 empirical / robust: {int(np.sum(empirical.stable_loading_mask_[0]))} / {int(np.sum(robust.stable_loading_mask_[0]))}")

    component = 0
    x = np.arange(maturities.size)
    fig = plt.figure(figsize=(9.2, 5.2))
    ax = fig.add_subplot(111)
    for analysis, label, offset in [
        (empirical, "empirical PCA", -0.08),
        (robust, "RobustPCA", 0.08),
    ]:
        center = analysis.components_[component]
        lower = analysis.loading_interval_lower_[component]
        upper = analysis.loading_interval_upper_[component]
        ax.errorbar(
            x + offset,
            center,
            yerr=np.vstack([center - lower, upper - center]),
            fmt="o",
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [f"{value:g}Y" for value in maturities], rotation=35, ha="right")
    ax.set_ylabel("loading")
    ax.set_title("First factor loading with 90% bootstrap intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loading_intervals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8.2, 4.8))
    ax = fig.add_subplot(111)
    bins = np.linspace(
        0.0,
        max(
            np.max(empirical.max_principal_angle_degrees_),
            np.max(robust.max_principal_angle_degrees_),
        ) * 1.05,
        18,
    )
    ax.hist(empirical.max_principal_angle_degrees_, bins=bins, alpha=0.55, label="empirical PCA")
    ax.hist(robust.max_principal_angle_degrees_, bins=bins, alpha=0.55, label="RobustPCA")
    ax.set_xlabel("largest principal angle to full-data subspace (degrees)")
    ax.set_ylabel("bootstrap count")
    ax.set_title("Bootstrap variation of the retained factor subspace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "principal_angle_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    components = np.arange(1, 4)
    fig = plt.figure(figsize=(8.2, 4.8))
    ax = fig.add_subplot(111)
    for analysis, label, offset in [
        (empirical, "empirical PCA", -0.08),
        (robust, "RobustPCA", 0.08),
    ]:
        center = analysis.eigenvalues_
        lower = analysis.eigenvalue_interval_lower_
        upper = analysis.eigenvalue_interval_upper_
        ax.errorbar(
            components + offset,
            center,
            yerr=np.vstack([center - lower, upper - center]),
            fmt="o",
            capsize=4,
            label=label,
        )
    ax.set_xticks(components)
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Bootstrap uncertainty in robust factor variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "eigenvalue_intervals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (outdir / "metrics.csv").write_text(
        "metric,empirical,robust\n"
        f"median_max_principal_angle_degrees,{empirical.median_max_principal_angle_degrees_:.8f},{robust.median_max_principal_angle_degrees_:.8f}\n"
        f"upper_max_principal_angle_degrees,{empirical.max_principal_angle_interval_degrees_[1]:.8f},{robust.max_principal_angle_interval_degrees_[1]:.8f}\n"
        f"reference_subspace_error,{empirical_error:.8f},{robust_error:.8f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
