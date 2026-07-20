"""Classical and robust SOBI under impulsive time-series contamination."""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _align_sources(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference_standardized = (reference - reference.mean(axis=0)) / reference.std(axis=0)
    estimate_standardized = (estimate - estimate.mean(axis=0)) / estimate.std(axis=0)
    correlation = reference_standardized.T @ estimate_standardized / reference.shape[0]
    best_permutation = max(
        permutations(range(reference.shape[1])),
        key=lambda order: sum(abs(correlation[index, order[index]]) for index in range(reference.shape[1])),
    )
    aligned = estimate[:, best_permutation].copy()
    for index in range(reference.shape[1]):
        denominator = float(aligned[:, index] @ aligned[:, index])
        aligned[:, index] *= float(aligned[:, index] @ reference[:, index]) / denominator
    return aligned


def _autocorrelation(series: np.ndarray, lags: np.ndarray) -> np.ndarray:
    centered = series - series.mean(axis=0)
    variance = np.mean(centered * centered, axis=0)
    values = []
    for lag in lags:
        values.append(np.mean(centered[:-lag] * centered[lag:], axis=0) / variance)
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/use_cases/sobi_source_separation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(11)
    n_samples = 3000
    mixing = np.array(
        [[1.0, 0.4, -0.2], [0.2, 1.2, 0.5], [-0.4, 0.3, 0.9]],
        dtype=float,
    )
    coefficients = np.array([0.90, -0.60, 0.25])
    innovations = rng.normal(size=(n_samples, 3))
    sources = np.zeros_like(innovations)
    for index in range(1, n_samples):
        sources[index] = coefficients * sources[index - 1] + innovations[index]

    observed = sources @ mixing.T
    contaminated = observed.copy()
    impulse_rows = rng.choice(n_samples, 55, replace=False)
    contaminated[impulse_rows] += rng.normal(scale=30.0, size=(impulse_rows.size, 3))

    classical = rc.SOBI(lags=15, backend="auto").fit(contaminated)
    robust = rc.RobustSOBI(lags=15, lag_weighting="huber", backend="auto").fit(contaminated)
    classical_aligned = _align_sources(sources, classical.transform(observed))
    robust_aligned = _align_sources(sources, robust.transform(observed))

    classical_mdi = rc.minimum_distance_index(classical.unmixing_, mixing)
    robust_mdi = rc.minimum_distance_index(robust.unmixing_, mixing)
    print(f"Classical SOBI MDI: {classical_mdi:.6f}")
    print(f"Robust SOBI MDI: {robust_mdi:.6f}")
    print(f"Classical off-diagonal energy: {classical.off_diagonal_energy_:.6f}")
    print(f"Robust off-diagonal energy: {robust.off_diagonal_energy_:.6f}")
    print(f"Temporal signatures: {robust.temporal_signatures_.shape}")

    center = int(np.sort(impulse_rows)[len(impulse_rows) // 2])
    start = max(0, center - 160)
    stop = min(n_samples, start + 420)
    time = np.arange(start, stop)
    fig = plt.figure(figsize=(10.5, 8.0))
    for index in range(3):
        ax = fig.add_subplot(3, 1, index + 1)
        ax.plot(time, sources[start:stop, index], linewidth=1.0, label="true source")
        ax.plot(time, classical_aligned[start:stop, index], linewidth=0.8, alpha=0.75, label="classical SOBI")
        ax.plot(time, robust_aligned[start:stop, index], linewidth=0.9, alpha=0.85, label="RobustSOBI")
        local_impulses = impulse_rows[(impulse_rows >= start) & (impulse_rows < stop)]
        for impulse in local_impulses:
            ax.axvline(impulse, linewidth=0.7, alpha=0.18)
        ax.set_ylabel(f"source {index + 1}")
        if index == 0:
            ax.legend(ncol=3)
    ax.set_xlabel("time index")
    fig.suptitle("SOBI source recovery around impulsive contamination")
    fig.tight_layout()
    fig.savefig(outdir / "source_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lags = np.arange(1, 16)
    true_acf = _autocorrelation(sources, lags)
    classical_acf = _autocorrelation(classical_aligned, lags)
    robust_acf = _autocorrelation(robust_aligned, lags)
    fig = plt.figure(figsize=(9.5, 7.5))
    for index in range(3):
        ax = fig.add_subplot(3, 1, index + 1)
        ax.plot(lags, true_acf[:, index], marker="o", label="true")
        ax.plot(lags, classical_acf[:, index], marker="s", label="classical")
        ax.plot(lags, robust_acf[:, index], marker="^", label="robust")
        ax.axhline(0.0, linewidth=0.7)
        ax.set_ylabel(f"source {index + 1}")
        if index == 0:
            ax.legend(ncol=3)
    ax.set_xlabel("lag")
    fig.suptitle("Recovered temporal correlation signatures")
    fig.tight_layout()
    fig.savefig(outdir / "lag_signatures.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(111)
    ax.bar(["Classical SOBI", "RobustSOBI"], [classical_mdi, robust_mdi])
    ax.set_ylabel("minimum-distance index")
    ax.set_title("Source-separation error under impulses")
    fig.tight_layout()
    fig.savefig(outdir / "mdi_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
