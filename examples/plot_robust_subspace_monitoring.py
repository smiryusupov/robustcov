# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Monitor a production embedding stream with a frozen robust reference.

The example is deterministic and entirely synthetic so it can run in CI and in
Sphinx without downloading a model.  It demonstrates four operating phases:
stable traffic, drift along a known representation direction, rotation of the
latent subspace, and a minority of out-of-subspace corrupted vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustcov as rc


def _orthonormal_basis(
    rng: np.random.Generator,
    n_features: int,
    rank: int,
) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    return basis[:, :rank]


def make_stream(seed: int = 123):
    rng = np.random.default_rng(seed)
    n_features = 20
    rank = 4
    basis = _orthonormal_basis(rng, n_features, rank + 1)
    reference_basis = basis[:, :rank]
    orthogonal_direction = basis[:, rank]
    scales = np.array([3.0, 2.2, 1.5, 0.9])

    def sample(n: int, active_basis: np.ndarray = reference_basis) -> np.ndarray:
        latent = rng.normal(size=(n, rank)) * scales
        return latent @ active_basis.T + rng.normal(
            scale=0.12,
            size=(n, n_features),
        )

    reference = sample(720)
    contaminated = rng.choice(reference.shape[0], size=30, replace=False)
    reference[contaminated] += rng.normal(
        7.0,
        0.5,
        size=(contaminated.size, 1),
    ) * orthogonal_direction

    theta = np.deg2rad(55.0)
    rotated_basis = reference_basis.copy()
    rotated_basis[:, -1] = (
        np.cos(theta) * reference_basis[:, -1]
        + np.sin(theta) * orthogonal_direction
    )

    batches: list[np.ndarray] = []
    phases: list[str] = []
    labels: list[np.ndarray] = []
    for batch_index in range(18):
        if batch_index < 5:
            X = sample(40)
            phase = "stable"
            outlier = np.zeros(40, dtype=bool)
        elif batch_index < 9:
            X = sample(40) + 2.8 * reference_basis[:, 0]
            phase = "location drift"
            outlier = np.zeros(40, dtype=bool)
        elif batch_index < 13:
            X = sample(40, rotated_basis)
            phase = "subspace rotation"
            outlier = np.zeros(40, dtype=bool)
        else:
            X = sample(40)
            outlier = np.zeros(40, dtype=bool)
            selected = rng.choice(40, size=10, replace=False)
            X[selected] += rng.normal(
                9.0,
                0.5,
                size=(selected.size, 1),
            ) * orthogonal_direction
            outlier[selected] = True
            phase = "OOD mixture"
        batches.append(X)
        phases.append(phase)
        labels.append(outlier)

    return reference, batches, np.asarray(phases), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results/use_cases/robust_subspace_monitoring",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference, batches, phases, labels = make_stream()
    monitor = rc.RobustSubspaceMonitor(
        n_components=4,
        estimator=rc.RegularizedCauchy(
            alpha=0.12,
            max_iter=80,
            tol=1e-5,
            warn_on_nonconvergence=False,
        ),
        window_size=120,
        calibration_windows=12,
        threshold_quantile=0.98,
        sample_quantile=0.95,
        threshold_scale=1.5,
        alarm_patience=2,
        random_state=7,
        history_size=30,
    ).fit(reference)

    results = [monitor.update(batch) for batch in batches]
    ready = [result for result in results if result.ready]
    ready_phases = phases[[result.ready for result in results]]

    print("Robust rolling subspace monitoring")
    print("==================================")
    print(f"reference shape: {reference.shape}")
    print(f"rolling window: {monitor.window_size}")
    print(f"retained components: {monitor.n_components_}")
    print(f"ready updates: {len(ready)} / {len(results)}")
    print(f"persistent alarms: {sum(result.alarm for result in ready)}")
    for phase in dict.fromkeys(ready_phases.tolist()):
        phase_results = [
            result
            for result, result_phase in zip(ready, ready_phases)
            if result_phase == phase
        ]
        location_ratio = np.median(
            [
                result.location_shift / result.thresholds["location_shift"]
                for result in phase_results
            ]
        )
        angle_ratio = np.median(
            [
                result.max_subspace_angle
                / result.thresholds["max_subspace_angle"]
                for result in phase_results
            ]
        )
        orthogonal_ratio = np.median(
            [
                result.orthogonal_distance_shift
                / result.thresholds["orthogonal_distance_shift"]
                for result in phase_results
            ]
        )
        outlier_fraction = np.median(
            [result.combined_outlier_fraction for result in phase_results]
        )
        print(
            f"{phase:18s} "
            f"location/threshold={location_ratio:5.2f}, "
            f"angle/threshold={angle_ratio:5.2f}, "
            f"orthogonal/threshold={orthogonal_ratio:5.2f}, "
            f"outlier_fraction={outlier_fraction:5.2f}"
        )

    rc.plot_subspace_monitor_history(
        monitor,
        title="Frozen-reference robust monitoring separates drift mechanisms",
        output_path=outdir / "monitor_history.png",
        show=False,
    )

    batch_index = len(batches) - 1
    final_result = results[batch_index]
    fig = plt.figure(figsize=(7.4, 5.2))
    ax = fig.add_subplot(111)
    known = labels[batch_index]
    ax.scatter(
        final_result.score_distances[~known],
        final_result.orthogonal_distances[~known],
        s=28,
        alpha=0.75,
        label="ordinary production vector",
    )
    ax.scatter(
        final_result.score_distances[known],
        final_result.orthogonal_distances[known],
        s=70,
        facecolors="none",
        edgecolors="black",
        label="injected out-of-subspace vector",
    )
    ax.axvline(monitor.score_distance_threshold_, linestyle="--")
    ax.axhline(monitor.orthogonal_distance_threshold_, linestyle="--")
    ax.set_xlabel("reference score distance")
    ax.set_ylabel("reference orthogonal distance")
    ax.set_title("Record-level diagnosis inside an alerted batch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "final_batch_outlier_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    normalized = []
    for result in ready:
        normalized.append(
            [
                result.location_shift / result.thresholds["location_shift"],
                result.scale_shift / result.thresholds["scale_shift"],
                result.shape_shift / result.thresholds["shape_shift"],
                result.max_subspace_angle
                / result.thresholds["max_subspace_angle"],
                result.orthogonal_distance_shift
                / result.thresholds["orthogonal_distance_shift"],
            ]
        )
    normalized = np.asarray(normalized)
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    image = ax.imshow(normalized.T, aspect="auto", vmin=0.0, vmax=3.0)
    ax.set_yticks(range(5))
    ax.set_yticklabels(
        ["location", "scale", "shape", "subspace angle", "orthogonal distance"]
    )
    ax.set_xticks(range(len(ready)))
    ax.set_xticklabels(ready_phases, rotation=55, ha="right", fontsize=8)
    ax.set_title("Drift mechanism map: each value is metric / calibrated threshold")
    fig.colorbar(image, ax=ax, label="threshold ratio (clipped at 3)")
    fig.tight_layout()
    fig.savefig(outdir / "drift_mechanism_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
