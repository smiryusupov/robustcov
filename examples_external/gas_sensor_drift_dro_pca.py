#!/usr/bin/env python
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Evaluate DRO-PCA on the UCI gas-sensor temporal-drift batches.

The script keeps all raw data in the robustcov user cache.  It residualizes gas
identity and log concentration using early batches, learns a transport geometry
from early temporal batch shifts, calibrates window alerts on held-out early
batches, and evaluates later batches.  A separately labelled synthetic sensor
failure is included as an off-geometry stress control.

Examples
--------
Fetch and run::

    python examples_external/gas_sensor_drift_dro_pca.py --download

Use an existing cache::

    python examples_external/gas_sensor_drift_dro_pca.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robustcov.datasets import DatasetNotFoundError, fetch_gas_sensor_drift
from robustcov.experimental import DistributionallyRobustPCA

from examples_external._dro_pca_utils import (
    diagonal_transport_from_domain_means,
    empirical_pca,
    reconstruction_errors,
    robust_standardization,
    upper_order_statistic,
    window_means,
)


def residualize_known_conditions(
    X: np.ndarray,
    gas: np.ndarray,
    concentration: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Remove early-batch gas identity and log-concentration effects."""

    X = np.asarray(X, dtype=float)
    gas = np.asarray(gas, dtype=int)
    log_concentration = np.log1p(np.asarray(concentration, dtype=float))
    residual = np.empty_like(X)
    for gas_id in np.unique(gas):
        selected = fit_mask & (gas == gas_id)
        if int(np.sum(selected)) < 3:
            raise ValueError(f"gas code {gas_id} has too few early-batch observations")
        design_fit = np.column_stack((np.ones(int(np.sum(selected))), log_concentration[selected]))
        coefficients, *_ = np.linalg.lstsq(design_fit, X[selected], rcond=None)
        target = gas == gas_id
        design_all = np.column_stack((np.ones(int(np.sum(target))), log_concentration[target]))
        residual[target] = X[target] - design_all @ coefficients
    return residual


def sensor_failure_control(X: np.ndarray, *, seed: int = 20260720) -> np.ndarray:
    """Inject an explicitly synthetic failure into two 8-feature sensor blocks."""

    rng = np.random.default_rng(seed)
    failed = np.asarray(X, dtype=float).copy()
    for sensor in (3, 12):
        block = slice(sensor * 8, (sensor + 1) * 8)
        failed[:, block] = 1.8 * failed[:, block] + 2.5
    failed += rng.normal(scale=0.03, size=failed.shape)
    return failed


def run_analysis(
    X: np.ndarray,
    gas: np.ndarray,
    concentration: np.ndarray,
    batch: np.ndarray,
    *,
    n_components: int = 12,
    false_alarm_rate: float = 0.05,
    radius_scale: float = 2.0,
    window_size: int = 64,
    window_step: int = 32,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    fit_mask = batch <= 3
    calibration_mask = (batch >= 4) & (batch <= 5)
    if not np.any(fit_mask) or not np.any(calibration_mask):
        raise ValueError("analysis requires training batches 1-3 and calibration batches 4-5")

    residual = residualize_known_conditions(X, gas, concentration, fit_mask)
    standardization = robust_standardization(residual[fit_mask])
    Z = standardization.transform(residual)
    rank = min(int(n_components), Z.shape[1] - 1)
    if rank < 1:
        raise ValueError("n_components is incompatible with the retained features")

    transport = diagonal_transport_from_domain_means(
        Z[calibration_mask], batch[calibration_mask], ridge_fraction=0.05
    )
    empirical_location, empirical_basis = empirical_pca(Z[fit_mask], rank)
    dro = DistributionallyRobustPCA(
        n_components=rank,
        radius="sqrt_n",
        radius_scale=radius_scale,
        transport_matrix=transport,
        formulation="exact",
    ).fit(Z[fit_mask])

    empirical_errors = reconstruction_errors(Z, empirical_location, empirical_basis)
    dro_errors = dro.reconstruction_error(Z)
    calibration_empirical: list[float] = []
    calibration_dro: list[float] = []
    for batch_id in (4, 5):
        selected = batch == batch_id
        calibration_empirical.extend(window_means(empirical_errors[selected], window_size, window_step))
        calibration_dro.extend(window_means(dro_errors[selected], window_size, window_step))
    thresholds = {
        "Empirical PCA": upper_order_statistic(np.asarray(calibration_empirical), false_alarm_rate),
        "DRO-PCA": upper_order_statistic(np.asarray(calibration_dro), false_alarm_rate),
    }

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for batch_id in range(1, 11):
        selected = batch == batch_id
        scores = {
            "Empirical PCA": window_means(empirical_errors[selected], window_size, window_step),
            "DRO-PCA": window_means(dro_errors[selected], window_size, window_step),
        }
        for method, values in scores.items():
            threshold = thresholds[method]
            for window_id, value in enumerate(values, start=1):
                rows.append(
                    {
                        "scenario": "observed_temporal_batch",
                        "batch": batch_id,
                        "window": window_id,
                        "method": method,
                        "mean_reconstruction_risk": float(value),
                        "threshold": threshold,
                        "normalized_risk": float(value / threshold),
                        "alert": int(value > threshold),
                    }
                )
            summary.append(
                {
                    "scenario": "observed_temporal_batch",
                    "batch": batch_id,
                    "method": method,
                    "mean_risk": float(np.mean(values)),
                    "alert_rate": float(np.mean(values > threshold)),
                    "n_windows": int(values.size),
                }
            )

    late = batch == 10
    failed_residual = sensor_failure_control(residual[late])
    failed_Z = standardization.transform(failed_residual)
    failure_scores = {
        "Empirical PCA": window_means(
            reconstruction_errors(failed_Z, empirical_location, empirical_basis),
            window_size,
            window_step,
        ),
        "DRO-PCA": window_means(dro.reconstruction_error(failed_Z), window_size, window_step),
    }
    for method, values in failure_scores.items():
        threshold = thresholds[method]
        for window_id, value in enumerate(values, start=1):
            rows.append(
                {
                    "scenario": "synthetic_off_geometry_sensor_failure",
                    "batch": 10,
                    "window": window_id,
                    "method": method,
                    "mean_reconstruction_risk": float(value),
                    "threshold": threshold,
                    "normalized_risk": float(value / threshold),
                    "alert": int(value > threshold),
                }
            )
        summary.append(
            {
                "scenario": "synthetic_off_geometry_sensor_failure",
                "batch": 10,
                "method": method,
                "mean_risk": float(np.mean(values)),
                "alert_rate": float(np.mean(values > threshold)),
                "n_windows": int(values.size),
            }
        )

    metadata: dict[str, object] = {
        "fit_batches": "1-3",
        "calibration_batches": "4-5",
        "test_batches": "6-10",
        "n_components": rank,
        "n_features_retained": int(Z.shape[1]),
        "radius": float(dro.radius_),
        "radius_calibration": dro.radius_calibration_,
        "exact_worst_case_risk": float(dro.exact_worst_case_risk_),
        "transport_geometry": "batch_mean_shift_diagonal",
        "threshold": "finite_sample_calibration_window_order_statistic",
    }
    return rows, summary, metadata


def save_outputs(
    outdir: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("this external example requires robustcov[plot]") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "window_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    with (outdir / "run_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(metadata.items())

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for method in ("Empirical PCA", "DRO-PCA"):
        values = [
            row for row in summary
            if row["scenario"] == "observed_temporal_batch" and row["method"] == method
        ]
        ax.plot(
            [int(row["batch"]) for row in values],
            [float(row["mean_risk"]) for row in values],
            marker="o",
            label=method,
        )
    ax.axvspan(0.5, 3.5, alpha=0.08, label="Fit batches")
    ax.axvspan(3.5, 5.5, alpha=0.08, label="Calibration batches")
    ax.set_xlabel("Temporal batch")
    ax.set_ylabel("Mean window reconstruction risk")
    ax.set_title("UCI gas-sensor drift: held-out temporal reconstruction risk")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "batch_risk.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    width = 0.36
    batches = np.arange(1, 11)
    for offset, method in enumerate(("Empirical PCA", "DRO-PCA")):
        values = [
            float(next(
                row["alert_rate"] for row in summary
                if row["scenario"] == "observed_temporal_batch"
                and row["method"] == method
                and int(row["batch"]) == batch_id
            ))
            for batch_id in batches
        ]
        ax.bar(batches + (offset - 0.5) * width, values, width=width, label=method)
    ax.set_xticks(batches)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Temporal batch")
    ax.set_ylabel("Alert rate across windows")
    ax.set_title("Window alerts after independent early-batch calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "batch_alert_rates.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    failure = [row for row in summary if row["scenario"] == "synthetic_off_geometry_sensor_failure"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar([str(row["method"]) for row in failure], [float(row["alert_rate"]) for row in failure])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Alert rate")
    ax.set_title("Explicit synthetic off-geometry sensor-failure control")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "sensor_failure_control.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="explicitly download into the user cache")
    parser.add_argument("--archive", help="manually downloaded UCI ZIP archive")
    parser.add_argument("--cache-dir")
    parser.add_argument("--outdir", type=Path, default=Path("results/external/gas_sensor_drift_dro_pca"))
    parser.add_argument("--n-components", type=int, default=12)
    parser.add_argument("--radius-scale", type=float, default=2.0)
    args = parser.parse_args()

    try:
        dataset = fetch_gas_sensor_drift(
            cache_dir=args.cache_dir,
            download=args.download,
            archive_path=args.archive,
        )
    except DatasetNotFoundError as exc:
        print(exc)
        print("Download explicitly with --download or pass --archive /path/to/archive.zip")
        raise SystemExit(0)

    rows, summary, metadata = run_analysis(
        dataset.X,
        dataset.gas,
        dataset.concentration,
        dataset.batch,
        n_components=args.n_components,
        radius_scale=args.radius_scale,
    )
    metadata.update(
        {
            "dataset_homepage": dataset.info.homepage,
            "dataset_citation": dataset.info.citation,
            "archive_sha256": dataset.archive_sha256,
            "cache_dir": str(dataset.data_dir.parent),
        }
    )
    save_outputs(args.outdir, rows, summary, metadata)
    print("scenario,batch,method,mean_risk,alert_rate,n_windows")
    for row in summary:
        print(
            f"{row['scenario']},{row['batch']},{row['method']},"
            f"{float(row['mean_risk']):.6f},{float(row['alert_rate']):.3f},{row['n_windows']}"
        )
    print(f"saved,{args.outdir}")


if __name__ == "__main__":
    main()
