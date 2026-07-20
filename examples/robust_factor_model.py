"""Robust static factor estimation with automatic factor-number selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _subspace_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    left, _ = np.linalg.qr(estimated)
    right, _ = np.linalg.qr(truth)
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.sqrt(np.mean(1.0 - np.clip(singular_values, 0.0, 1.0) ** 2)))


def _align_factorization(
    estimated_loadings: np.ndarray,
    estimated_scores: np.ndarray,
    true_loadings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left, _, right_t = np.linalg.svd(estimated_loadings.T @ true_loadings, full_matrices=False)
    rotation = left @ right_t
    return estimated_loadings @ rotation, estimated_scores @ rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/use_cases/robust_factor_model")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(19)
    n_samples, n_features, n_factors = 700, 18, 3
    loadings, _ = np.linalg.qr(rng.normal(size=(n_features, n_factors)))
    factors = rng.standard_t(4, size=(n_samples, n_factors))
    clean_data = factors @ loadings.T + 0.22 * rng.normal(size=(n_samples, n_features))
    data = clean_data.copy()

    bad_rows = rng.choice(n_samples, 35, replace=False)
    data[bad_rows] += rng.normal(scale=9.0, size=(bad_rows.size, n_features))

    model = rc.RobustFactorModel(
        n_factors="auto",
        method="kendall",
        max_factors=7,
    ).fit(data)

    reconstruction_error = np.linalg.norm(data - model.common_component_ - model.idiosyncratic_) / np.linalg.norm(data)
    subspace_error = _subspace_error(model.loadings_, loadings)
    aligned_loadings, aligned_scores = _align_factorization(
        model.loadings_, model.factor_scores_, loadings
    )
    print(f"Selected factor count: {model.n_factors_}")
    print(f"Loading-subspace error: {subspace_error:.6f}")
    print(f"Relative decomposition error: {reconstruction_error:.3e}")
    print(f"Factor scores: {model.factor_scores_.shape}")
    print(f"Common component: {model.common_component_.shape}")

    feature_index = np.arange(1, n_features + 1)
    fig = plt.figure(figsize=(9.5, 7.3))
    for index in range(n_factors):
        ax = fig.add_subplot(n_factors, 1, index + 1)
        ax.plot(feature_index, loadings[:, index], marker="o", label="true loading")
        ax.plot(feature_index, aligned_loadings[:, index], marker="s", label="estimated loading")
        ax.set_ylabel(f"factor {index + 1}")
        if index == 0:
            ax.legend(ncol=2)
    ax.set_xlabel("feature")
    fig.suptitle("Robust factor loading-subspace recovery")
    fig.tight_layout()
    fig.savefig(outdir / "loading_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    display = slice(0, 260)
    fig = plt.figure(figsize=(10, 7.3))
    for index in range(n_factors):
        ax = fig.add_subplot(n_factors, 1, index + 1)
        ax.plot(factors[display, index], linewidth=1.0, label="true factor")
        ax.plot(aligned_scores[display, index], linewidth=0.9, alpha=0.8, label="estimated factor")
        ax.set_ylabel(f"factor {index + 1}")
        if index == 0:
            ax.legend(ncol=2)
    ax.set_xlabel("sample")
    fig.suptitle("Robust factor-score recovery")
    fig.tight_layout()
    fig.savefig(outdir / "factor_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    eigenvalues = model.kendall_eigenvalues_
    candidate_count = min(8, eigenvalues.size)
    fig = plt.figure(figsize=(8.2, 4.5))
    ax = fig.add_subplot(111)
    positions = np.arange(1, candidate_count + 1)
    ax.plot(positions, eigenvalues[:candidate_count], marker="o")
    ax.axvline(model.n_factors_, linestyle="--", label=f"selected: {model.n_factors_}")
    ax.set_xlabel("ordered component")
    ax.set_ylabel("spatial-Kendall eigenvalue")
    ax.set_title("Automatic factor-count selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "factor_selection.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
