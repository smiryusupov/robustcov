"""Robust two-scatter ICA on a contaminated linear mixture."""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _align_sources(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align estimated sources to truth for visualization only."""

    reference_standardized = (reference - reference.mean(axis=0)) / reference.std(axis=0)
    estimate_standardized = (estimate - estimate.mean(axis=0)) / estimate.std(axis=0)
    correlation = reference_standardized.T @ estimate_standardized / reference.shape[0]
    best_permutation = max(
        permutations(range(reference.shape[1])),
        key=lambda order: sum(abs(correlation[index, order[index]]) for index in range(reference.shape[1])),
    )
    aligned = estimate[:, best_permutation].copy()
    component_correlations = np.empty(reference.shape[1], dtype=float)
    for index in range(reference.shape[1]):
        denominator = float(aligned[:, index] @ aligned[:, index])
        scale = float(aligned[:, index] @ reference[:, index]) / denominator
        aligned[:, index] *= scale
        component_correlations[index] = abs(np.corrcoef(reference[:, index], aligned[:, index])[0, 1])
    return aligned, component_correlations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/use_cases/ica_two_scatter")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    n_samples = 2500
    mixing = np.array(
        [[1.0, 0.4, -0.2], [0.2, 1.2, 0.5], [-0.4, 0.3, 0.9]],
        dtype=float,
    )
    sources = np.column_stack(
        [
            rng.laplace(size=n_samples) / np.sqrt(2.0),
            rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples),
            rng.standard_t(6, size=n_samples) / np.sqrt(1.5),
        ]
    )
    observed = sources @ mixing.T

    contaminated = observed.copy()
    bad_rows = rng.choice(n_samples, 35, replace=False)
    contaminated[bad_rows] += rng.normal(scale=18.0, size=(bad_rows.size, 3))

    model = rc.TwoScatterICA(
        radial_clip_quantile=0.90,
        random_state=0,
    ).fit(contaminated)

    recovered = model.transform(observed)
    recovered_aligned, correlations = _align_sources(sources, recovered)
    reconstructed = model.inverse_transform(recovered)
    relative_reconstruction_error = np.linalg.norm(reconstructed - observed) / np.linalg.norm(observed)

    print(f"Minimum-distance index: {rc.minimum_distance_index(model.unmixing_, mixing):.6f}")
    print(f"Amari index: {rc.amari_index(model.unmixing_, mixing):.6f}")
    print(f"Relative reconstruction error: {relative_reconstruction_error:.3e}")
    print(f"Recovered source matrix: {recovered.shape}")
    print("Absolute source correlations: " + ", ".join(f"{value:.3f}" for value in correlations))

    display = slice(0, 450)
    fig = plt.figure(figsize=(10, 7.5))
    for index in range(3):
        ax = fig.add_subplot(3, 1, index + 1)
        ax.plot(sources[display, index], linewidth=1.0, label="true source")
        ax.plot(recovered_aligned[display, index], linewidth=0.9, alpha=0.8, label="recovered")
        ax.set_ylabel(f"source {index + 1}")
        if index == 0:
            ax.legend(ncol=2)
    ax.set_xlabel("sample")
    fig.suptitle("Two-scatter ICA source recovery")
    fig.tight_layout()
    fig.savefig(outdir / "source_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.4))
    ax = fig.add_subplot(1, 2, 1)
    clean_mask = np.ones(n_samples, dtype=bool)
    clean_mask[bad_rows] = False
    ax.scatter(contaminated[clean_mask, 0], contaminated[clean_mask, 1], s=8, alpha=0.35, label="ordinary rows")
    ax.scatter(contaminated[bad_rows, 0], contaminated[bad_rows, 1], s=24, marker="x", label="contaminated rows")
    ax.set_xlabel("observed channel 1")
    ax.set_ylabel("observed channel 2")
    ax.set_title("Contaminated mixture")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(recovered_aligned[:, 0], recovered_aligned[:, 1], s=8, alpha=0.35)
    ax.set_xlabel("recovered source 1")
    ax.set_ylabel("recovered source 2")
    ax.set_title("Recovered independent coordinates")
    fig.tight_layout()
    fig.savefig(outdir / "mixture_and_sources.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
