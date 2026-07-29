# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Compare IID and stationary bootstrap uncertainty for serially dependent factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate heavy-tailed, serially dependent multivariate factor data."""
    rng = np.random.default_rng(seed)
    n_samples, n_features, n_factors = 320, 8, 2

    basis, _ = np.linalg.qr(rng.normal(size=(n_features, n_factors)))
    innovation_covariance = np.array([[1.0, 0.65], [0.65, 1.0]])
    innovation_cholesky = np.linalg.cholesky(innovation_covariance)
    factor_innovations = (
        rng.standard_t(df=5, size=(n_samples, n_factors))
        @ innovation_cholesky.T
    )

    factors = np.zeros((n_samples, n_factors))
    for time in range(1, n_samples):
        factors[time] = 0.94 * factors[time - 1] + factor_innovations[time]

    idiosyncratic = np.zeros((n_samples, n_features))
    idiosyncratic_innovations = rng.normal(
        scale=0.25,
        size=(n_samples, n_features),
    )
    for time in range(1, n_samples):
        idiosyncratic[time] = (
            0.60 * idiosyncratic[time - 1]
            + idiosyncratic_innovations[time]
        )

    X = factors @ np.diag([2.2, 1.2]) @ basis.T + idiosyncratic

    # Short bursts of feature-specific contamination. Robust scatter handles the
    # extreme rows; dependent resampling addresses the serial structure.
    for start in (80, 185, 250):
        X[start : start + 3, 5:] += rng.normal(
            loc=5.0,
            scale=1.0,
            size=(3, n_features - 5),
        )

    feature_names = np.array([f"series {index + 1}" for index in range(n_features)])
    return X, basis.T, feature_names


def lag_one_autocorrelation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results/use_cases/robust_pca_dependent_stability",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, true_components, feature_names = make_data()
    pca = rc.RobustPCA(
        n_components=2,
        estimator=rc.RegularizedCauchy(
            alpha=0.10,
            max_iter=120,
            tol=1e-5,
            warn_on_nonconvergence=False,
        ),
        store_scores=False,
    )

    common = dict(
        pca=pca,
        n_resamples=80,
        confidence_level=0.90,
        alignment="procrustes",
        random_state=3,
        min_successful_resamples=70,
    )
    iid = rc.SubspaceStability(resampling="iid", **common).fit(X)
    stationary = rc.SubspaceStability(
        resampling="stationary",
        block_length=16,
        **common,
    ).fit(X)

    reference_scores = stationary.reference_model_.transform(X)
    lag_one = lag_one_autocorrelation(reference_scores[:, 0])
    iid_loading_width = float(
        np.mean(iid.loading_interval_upper_ - iid.loading_interval_lower_)
    )
    stationary_loading_width = float(
        np.mean(
            stationary.loading_interval_upper_
            - stationary.loading_interval_lower_
        )
    )
    iid_subspace_error = float(
        np.linalg.norm(
            iid.components_.T @ iid.components_
            - true_components.T @ true_components,
            ord="fro",
        )
    )

    print("Dependent bootstrap stability for robust PCA")
    print("===========================================")
    print(f"observations: {X.shape[0]}")
    print(f"features: {X.shape[1]}")
    print(f"lag-1 autocorrelation of first robust score: {lag_one:.3f}")
    print(f"stationary-bootstrap expected block length: {stationary.block_length_}")
    print(
        "median largest principal angle, IID / stationary: "
        f"{iid.median_max_principal_angle_degrees_:.3f} / "
        f"{stationary.median_max_principal_angle_degrees_:.3f} degrees"
    )
    print(
        "90% upper largest angle, IID / stationary: "
        f"{iid.max_principal_angle_interval_degrees_[1]:.3f} / "
        f"{stationary.max_principal_angle_interval_degrees_[1]:.3f} degrees"
    )
    print(
        "mean loading-interval width, IID / stationary: "
        f"{iid_loading_width:.4f} / {stationary_loading_width:.4f}"
    )
    print(
        "first-eigenvalue standard error, IID / stationary: "
        f"{iid.eigenvalue_standard_error_[0]:.3f} / "
        f"{stationary.eigenvalue_standard_error_[0]:.3f}"
    )
    print(f"reference robust-subspace error: {iid_subspace_error:.3f}")

    component = 0
    positions = np.arange(X.shape[1])
    fig = plt.figure(figsize=(9.2, 5.2))
    ax = fig.add_subplot(111)
    for analysis, label, offset in [
        (iid, "IID bootstrap", -0.08),
        (stationary, "stationary bootstrap", 0.08),
    ]:
        center = analysis.components_[component]
        lower = analysis.loading_interval_lower_[component]
        upper = analysis.loading_interval_upper_[component]
        ax.errorbar(
            positions + offset,
            center,
            yerr=np.vstack([center - lower, upper - center]),
            fmt="o",
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, feature_names, rotation=35, ha="right")
    ax.set_ylabel("loading")
    ax.set_title("First loading: IID resampling understates uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loading_intervals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8.2, 4.8))
    ax = fig.add_subplot(111)
    upper = max(
        np.max(iid.max_principal_angle_degrees_),
        np.max(stationary.max_principal_angle_degrees_),
    )
    bins = np.linspace(0.0, upper * 1.05, 18)
    ax.hist(
        iid.max_principal_angle_degrees_,
        bins=bins,
        alpha=0.55,
        label="IID bootstrap",
    )
    ax.hist(
        stationary.max_principal_angle_degrees_,
        bins=bins,
        alpha=0.55,
        label="stationary bootstrap",
    )
    ax.set_xlabel("largest principal angle to full-data subspace (degrees)")
    ax.set_ylabel("bootstrap count")
    ax.set_title("Dependence-aware resampling broadens subspace uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "principal_angle_distribution.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(111)
    component_positions = np.arange(1, 3)
    width = 0.34
    ax.bar(
        component_positions - width / 2,
        iid.eigenvalue_standard_error_,
        width=width,
        label="IID bootstrap",
    )
    ax.bar(
        component_positions + width / 2,
        stationary.eigenvalue_standard_error_,
        width=width,
        label="stationary bootstrap",
    )
    ax.set_xticks(component_positions)
    ax.set_xlabel("component")
    ax.set_ylabel("bootstrap standard error")
    ax.set_title("Eigenvalue uncertainty under serial dependence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "eigenvalue_uncertainty.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (outdir / "metrics.csv").write_text(
        "metric,iid,stationary\n"
        f"median_max_principal_angle_degrees,{iid.median_max_principal_angle_degrees_:.8f},{stationary.median_max_principal_angle_degrees_:.8f}\n"
        f"upper_max_principal_angle_degrees,{iid.max_principal_angle_interval_degrees_[1]:.8f},{stationary.max_principal_angle_interval_degrees_[1]:.8f}\n"
        f"mean_loading_interval_width,{iid_loading_width:.8f},{stationary_loading_width:.8f}\n"
        f"first_eigenvalue_standard_error,{iid.eigenvalue_standard_error_[0]:.8f},{stationary.eigenvalue_standard_error_[0]:.8f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
