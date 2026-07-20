#!/usr/bin/env python
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Monitor C-MAPSS degradation while tolerating operating-condition shifts.

The default FD002 subset has six operating conditions and one fault mode.  The
workflow fits on early-life engine cycles, estimates a diagonal Wasserstein
transport geometry from healthy operating-regime differences, calibrates alerts
on independent early-life windows, and evaluates alert behavior over normalized
engine life.

Examples
--------
Fetch and run FD002::

    python examples_external/cmapss_dro_pca_monitoring.py --download

Use a manually downloaded NASA archive::

    python examples_external/cmapss_dro_pca_monitoring.py \
      --archive /path/to/CMAPSSData.zip
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

from robustcov import ConformalAlertCalibrator
from robustcov.datasets import DatasetNotFoundError, fetch_cmapss
from robustcov.experimental import DistributionallyRobustPCA

from examples_external._dro_pca_utils import (
    diagonal_transport_from_domain_means,
    empirical_pca,
    reconstruction_errors,
    robust_standardization,
)


def relative_life(unit: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit, dtype=int)
    cycle = np.asarray(cycle, dtype=float)
    maxima = {int(identifier): float(np.max(cycle[unit == identifier])) for identifier in np.unique(unit)}
    return np.asarray([value / maxima[int(identifier)] for identifier, value in zip(unit, cycle)], dtype=float)


def operating_regimes(settings: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Create deterministic coarse operating-condition domains from settings."""

    settings = np.asarray(settings, dtype=float)
    fit = settings[fit_mask]
    codes = np.zeros(settings.shape[0], dtype=np.int32)
    multiplier = 1
    for column in range(settings.shape[1]):
        values = fit[:, column]
        spread = float(np.ptp(values))
        scale = max(float(np.max(np.abs(values))), 1.0)
        if spread <= 1e-10 * scale:
            continue
        edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        digitized = np.digitize(settings[:, column], edges, right=False)
        codes += multiplier * digitized.astype(np.int32)
        multiplier *= len(edges) + 1
    return codes


def iter_engine_windows(
    values: np.ndarray,
    unit: np.ndarray,
    life: np.ndarray,
    *,
    window_size: int,
    step: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for engine in np.unique(unit):
        selected = np.flatnonzero(unit == engine)
        for start in range(0, max(selected.size - window_size + 1, 0), step):
            indices = selected[start : start + window_size]
            rows.append(
                {
                    "unit": int(engine),
                    "start_cycle": start + 1,
                    "life_midpoint": float(np.mean(life[indices])),
                    "indices": indices,
                    "mean_value": float(np.mean(values[indices])),
                }
            )
    return rows


def run_analysis(
    sensors: np.ndarray,
    settings: np.ndarray,
    unit: np.ndarray,
    cycle: np.ndarray,
    *,
    n_components: int = 6,
    radius_scale: float = 2.0,
    false_alarm_rate: float = 0.05,
    window_size: int = 30,
    step: int = 10,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], np.ndarray]:
    life = relative_life(unit, cycle)
    reference_mask = life <= 0.20
    calibration_mask = (life > 0.20) & (life <= 0.35)
    healthy_geometry_mask = life <= 0.35
    if int(np.sum(reference_mask)) < 100 or int(np.sum(calibration_mask)) < 100:
        raise ValueError("not enough early-life C-MAPSS rows for fitting and calibration")

    standardization = robust_standardization(sensors[reference_mask])
    Z = standardization.transform(sensors)
    rank = min(int(n_components), Z.shape[1] - 1)
    regimes = operating_regimes(settings, healthy_geometry_mask)
    healthy_regimes = regimes[healthy_geometry_mask]
    counts = {int(code): int(np.sum(healthy_regimes == code)) for code in np.unique(healthy_regimes)}
    retained_codes = {code for code, count in counts.items() if count >= 30}
    geometry_mask = healthy_geometry_mask & np.isin(regimes, tuple(retained_codes))
    if len(retained_codes) < 2:
        raise ValueError("fewer than two sufficiently populated operating regimes were found")
    transport = diagonal_transport_from_domain_means(
        Z[geometry_mask], regimes[geometry_mask], ridge_fraction=0.05
    )

    empirical_location, empirical_basis = empirical_pca(Z[reference_mask], rank)
    dro = DistributionallyRobustPCA(
        n_components=rank,
        radius="sqrt_n",
        radius_scale=radius_scale,
        transport_matrix=transport,
        formulation="exact",
    ).fit(Z[reference_mask])
    model_errors = {
        "Empirical PCA": reconstruction_errors(Z, empirical_location, empirical_basis),
        "DRO-PCA": dro.reconstruction_error(Z),
    }

    calibration_scores: dict[str, list[float]] = {method: [] for method in model_errors}
    all_windows: dict[str, list[dict[str, object]]] = {}
    for method, errors in model_errors.items():
        windows = iter_engine_windows(
            errors, unit, life, window_size=window_size, step=step
        )
        all_windows[method] = windows
        calibration_scores[method] = [
            float(row["mean_value"])
            for row in windows
            if 0.20 < float(row["life_midpoint"]) <= 0.35
        ]
        if not calibration_scores[method]:
            raise ValueError("no calibration windows were produced; reduce window_size")
    calibrators = {
        method: ConformalAlertCalibrator(
            alpha=false_alarm_rate,
            tail="upper",
        ).fit(np.asarray(values, dtype=float))
        for method, values in calibration_scores.items()
    }

    rows: list[dict[str, object]] = []
    for method, windows in all_windows.items():
        calibrator = calibrators[method]
        threshold = calibrator.threshold_
        for row in windows:
            risk = float(row["mean_value"])
            p_value = float(calibrator.p_values(risk))
            rows.append(
                {
                    "method": method,
                    "unit": int(row["unit"]),
                    "start_cycle": int(row["start_cycle"]),
                    "life_midpoint": float(row["life_midpoint"]),
                    "mean_reconstruction_risk": risk,
                    "conformal_p_value": p_value,
                    "threshold": threshold,
                    "normalized_risk": risk / threshold,
                    "alert": int(p_value <= false_alarm_rate),
                }
            )

    life_bins = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
    summary: list[dict[str, object]] = []
    for method in model_errors:
        selected_method = [row for row in rows if row["method"] == method]
        for lower, upper in life_bins:
            selected = [
                row for row in selected_method
                if lower < float(row["life_midpoint"]) <= upper
                or (lower == 0.0 and 0.0 <= float(row["life_midpoint"]) <= upper)
            ]
            if not selected:
                continue
            summary.append(
                {
                    "method": method,
                    "life_bin": f"{lower:.1f}-{min(upper, 1.0):.1f}",
                    "mean_risk": float(np.mean([float(row["mean_reconstruction_risk"]) for row in selected])),
                    "alert_rate": float(np.mean([int(row["alert"]) for row in selected])),
                    "n_windows": len(selected),
                }
            )

    late = life >= 0.85
    late_scores = dro.transform(Z[late])
    late_residual = Z[late] - dro.inverse_transform(late_scores)
    feature_contributions = np.mean(np.square(late_residual), axis=0)
    metadata: dict[str, object] = {
        "reference_life_fraction": "<=0.20",
        "calibration_life_fraction": "(0.20,0.35]",
        "n_operating_regimes": len(retained_codes),
        "n_components": rank,
        "n_sensors_retained": int(Z.shape[1]),
        "radius": float(dro.radius_),
        "exact_worst_case_risk": float(dro.exact_worst_case_risk_),
        "transport_geometry": "healthy_operating_regime_mean_shift_diagonal",
        "threshold": "split_conformal_upper_tail_p_value",
        "false_alarm_rate": float(false_alarm_rate),
        "calibration_p_value_resolution": {
            method: float(calibrator.min_p_value_)
            for method, calibrator in calibrators.items()
        },
    }
    return rows, summary, metadata, feature_contributions


def save_outputs(
    outdir: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    metadata: dict[str, object],
    feature_contributions: np.ndarray,
    sensor_names: tuple[str, ...],
    kept_features: np.ndarray | None = None,
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
        method_rows = [row for row in rows if row["method"] == method]
        life = np.asarray([float(row["life_midpoint"]) for row in method_rows])
        risk = np.asarray([float(row["normalized_risk"]) for row in method_rows])
        bins = np.linspace(0.0, 1.0, 21)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = [float(np.mean(risk[(life >= lo) & (life < hi)])) if np.any((life >= lo) & (life < hi)) else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(centers, means, marker="o", label=method)
    ax.axhline(1.0, linestyle="--", linewidth=1.4, label="Calibrated threshold")
    ax.axvspan(0.0, 0.2, alpha=0.08, label="Reference")
    ax.axvspan(0.2, 0.35, alpha=0.08, label="Calibration")
    ax.set_xlabel("Normalized engine life")
    ax.set_ylabel("Mean window risk / threshold")
    ax.set_title("C-MAPSS degradation monitoring across operating conditions")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "risk_over_engine_life.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    life_labels = list(dict.fromkeys(str(row["life_bin"]) for row in summary))
    x = np.arange(len(life_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for offset, method in enumerate(("Empirical PCA", "DRO-PCA")):
        values = [
            float(next(row["alert_rate"] for row in summary if row["method"] == method and row["life_bin"] == label))
            for label in life_labels
        ]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=method)
    ax.set_xticks(x, life_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Normalized-life bin")
    ax.set_ylabel("Alert rate")
    ax.set_title("Alert rates rise as run-to-failure trajectories degrade")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "alert_rate_by_life.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    names = np.asarray(sensor_names, dtype=object)
    if kept_features is not None:
        names = names[np.asarray(kept_features, dtype=bool)]
    order = np.argsort(feature_contributions)[::-1][: min(12, feature_contributions.size)]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.barh(np.arange(order.size), feature_contributions[order][::-1])
    ax.set_yticks(np.arange(order.size), names[order][::-1])
    ax.set_xlabel("Mean late-life squared residual contribution")
    ax.set_title("Sensors contributing most to late-life off-subspace risk")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "late_life_sensor_contributions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="explicitly download into the user cache")
    parser.add_argument("--archive", help="manually downloaded NASA ZIP archive")
    parser.add_argument("--cache-dir")
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD002")
    parser.add_argument("--outdir", type=Path, default=Path("results/external/cmapss_dro_pca_monitoring"))
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--radius-scale", type=float, default=2.0)
    args = parser.parse_args()

    try:
        dataset = fetch_cmapss(
            args.subset,
            cache_dir=args.cache_dir,
            download=args.download,
            archive_path=args.archive,
        )
    except DatasetNotFoundError as exc:
        print(exc)
        print("Download explicitly with --download or pass --archive /path/to/archive.zip")
        raise SystemExit(0)

    standardization = robust_standardization(dataset.train.sensors[relative_life(dataset.train.unit, dataset.train.cycle) <= 0.20])
    rows, summary, metadata, contributions = run_analysis(
        dataset.train.sensors,
        dataset.train.settings,
        dataset.train.unit,
        dataset.train.cycle,
        n_components=args.n_components,
        radius_scale=args.radius_scale,
    )
    metadata.update(
        {
            "dataset_subset": dataset.subset,
            "dataset_homepage": dataset.info.homepage,
            "dataset_citation": dataset.info.citation,
            "archive_sha256": dataset.archive_sha256,
            "cache_dir": str(dataset.data_dir.parent),
        }
    )
    save_outputs(
        args.outdir,
        rows,
        summary,
        metadata,
        contributions,
        dataset.sensor_names,
        standardization.keep,
    )
    print("method,life_bin,mean_risk,alert_rate,n_windows")
    for row in summary:
        print(
            f"{row['method']},{row['life_bin']},{float(row['mean_risk']):.6f},"
            f"{float(row['alert_rate']):.3f},{row['n_windows']}"
        )
    print(f"saved,{args.outdir}")


if __name__ == "__main__":
    main()
