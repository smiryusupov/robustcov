#!/usr/bin/env python3
"""Deterministic validation for experimental online robust subspace tracking."""

from __future__ import annotations

import numpy as np

import robustcov as rc


def basis(angle: float, p: int = 10) -> np.ndarray:
    value = np.zeros((p, 2))
    value[0, 0] = 1.0
    value[1, 1] = np.cos(angle)
    value[2, 1] = np.sin(angle)
    return value


def sample(rng: np.random.Generator, n: int, angle: float) -> np.ndarray:
    target = basis(angle)
    latent = rng.normal(size=(n, 2))
    return latent @ target.T + rng.normal(scale=0.04, size=(n, target.shape[0]))


def projector_error(components: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            components.T @ components - target @ target.T,
            ord="fro",
        )
    )


def main() -> None:
    rng = np.random.default_rng(7)
    initial = sample(rng, 500, 0.0)
    tracker = rc.OnlineRobustSubspaceTracker(
        n_components=2,
        update_interval=50,
        buffer_size=200,
        adaptation_rate=0.8,
        max_update_angle=30.0,
    ).fit(initial)
    frozen = tracker.components_.copy()

    rejected = 0
    corrected = 0
    target = basis(0.0)
    for angle in np.linspace(0.05, 0.65, 18):
        batch = sample(rng, 50, float(angle))
        rows = rng.choice(batch.shape[0], size=5, replace=False)
        cols = rng.integers(0, batch.shape[1], size=5)
        batch[rows, cols] += rng.choice([-25.0, 25.0], size=5)
        batch[0] += 15.0
        result = tracker.update(batch)
        rejected += result.n_rejected
        corrected += result.n_cell_corrections
        target = basis(float(angle))

    adaptive = projector_error(tracker.components_, target)
    baseline = projector_error(frozen, target)
    print("metric,value")
    print(f"adaptive_projector_error,{adaptive:.6f}")
    print(f"frozen_projector_error,{baseline:.6f}")
    print(f"relative_error,{adaptive / baseline:.6f}")
    print(f"subspace_updates,{tracker.n_updates_}")
    print(f"rejected_rows,{rejected}")
    print(f"repaired_cells,{corrected}")
    if not adaptive < 0.6 * baseline:
        raise SystemExit("adaptive tracker did not improve enough over frozen PCA")
    if corrected < 20:
        raise SystemExit("sparse corruption repair was not exercised")


if __name__ == "__main__":
    main()
