"""Monitor structured data drift with distributionally robust PCA.

The DRO-PCA estimator supplies a shift-aware reference subspace.  A separate
window-level calibration step turns reconstruction risk into an alerting rule.
The exact Wasserstein worst-case risk is retained as a model diagnostic; it is
not treated as a hypothesis-test threshold.

The synthetic stream contains three regimes:

* nominal data matching the reference period;
* a geometry-aligned covariance shift that the DRO subspace is designed to
  tolerate; and
* an off-geometry covariance shift that should trigger an alert.

Run from the repository root::

    python examples/distributionally_robust_pca_drift_monitoring.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robustcov.experimental import DistributionallyRobustPCA


def _empirical_pca(X: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    location = np.mean(X, axis=0)
    centered = X - location
    covariance = centered.T @ centered / len(X)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1][:rank]]
    return location, basis


def _mean_reconstruction_risk(
    X: np.ndarray,
    location: np.ndarray,
    basis: np.ndarray,
) -> float:
    centered = X - location
    residual = centered - (centered @ basis) @ basis.T
    return float(np.mean(np.einsum("ij,ij->i", residual, residual)))


def _upper_calibration_quantile(values: np.ndarray, false_alarm_rate: float) -> float:
    """Return the finite-sample upper calibration quantile.

    This is the usual split-calibration order statistic.  It calibrates the
    window statistic empirically; it does not convert the DRO ambiguity-set
    expectation into a per-window probability guarantee.
    """

    values = np.sort(np.asarray(values, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("calibration values must be a non-empty one-dimensional array")
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie strictly between zero and one")
    rank = int(np.ceil((values.size + 1) * (1.0 - false_alarm_rate)))
    return float(values[min(max(rank, 1), values.size) - 1])


def _draw_window(
    rng: np.random.Generator,
    variances: np.ndarray,
    window_size: int,
) -> np.ndarray:
    return rng.normal(size=(window_size, len(variances))) * np.sqrt(variances)


def build_monitoring_problem(
    seed: int = 20260720,
    *,
    window_size: int = 80,
    n_calibration_windows: int = 80,
    windows_per_regime: int = 12,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, np.ndarray]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    reference_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    aligned_variances = np.array([4.5, 4.0, 8.0, 7.0, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
    off_geometry_variances = np.array([4.5, 4.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 8.0, 7.0])

    X_reference = _draw_window(rng, reference_variances, 400)
    calibration = [
        _draw_window(rng, reference_variances, window_size)
        for _ in range(n_calibration_windows)
    ]

    regimes: list[tuple[str, np.ndarray]] = []
    for name, variances in (
        ("nominal", reference_variances),
        ("geometry_aligned", aligned_variances),
        ("off_geometry", off_geometry_variances),
    ):
        regimes.extend(
            (name, _draw_window(rng, variances, window_size))
            for _ in range(windows_per_regime)
        )

    metadata = {
        "reference_variances": reference_variances,
        "geometry_aligned_variances": aligned_variances,
        "off_geometry_variances": off_geometry_variances,
    }
    return X_reference, calibration, regimes, metadata


def run_monitor(
    X_reference: np.ndarray,
    calibration_windows: list[np.ndarray],
    stream_windows: list[tuple[str, np.ndarray]],
    *,
    false_alarm_rate: float = 0.05,
) -> tuple[list[dict[str, object]], list[dict[str, object]], DistributionallyRobustPCA]:
    empirical_location, empirical_basis = _empirical_pca(X_reference, rank=2)
    dro = DistributionallyRobustPCA(
        n_components=2,
        radius=2.5,
        transport_geometry="residual",
        formulation="exact",
    ).fit(X_reference)

    methods = {
        "Empirical PCA": (
            empirical_location,
            empirical_basis,
            lambda X: _mean_reconstruction_risk(X, empirical_location, empirical_basis),
        ),
        "DRO-PCA": (
            dro.location_,
            dro.components_.T,
            lambda X: float(np.mean(dro.reconstruction_error(X))),
        ),
    }

    thresholds: dict[str, float] = {}
    for method, (_, _, scorer) in methods.items():
        calibration_scores = np.array([scorer(window) for window in calibration_windows])
        thresholds[method] = _upper_calibration_quantile(
            calibration_scores,
            false_alarm_rate,
        )

    rows: list[dict[str, object]] = []
    for index, (regime, window) in enumerate(stream_windows, start=1):
        row: dict[str, object] = {"window": index, "regime": regime}
        for method, (_, _, scorer) in methods.items():
            risk = scorer(window)
            threshold = thresholds[method]
            key = "empirical" if method == "Empirical PCA" else "dro"
            row[f"{key}_risk"] = risk
            row[f"{key}_threshold"] = threshold
            row[f"{key}_normalized"] = risk / threshold
            row[f"{key}_alert"] = int(risk > threshold)
        rows.append(row)

    summary: list[dict[str, object]] = []
    for method, key in (("Empirical PCA", "empirical"), ("DRO-PCA", "dro")):
        for regime in ("nominal", "geometry_aligned", "off_geometry"):
            selected = [row for row in rows if row["regime"] == regime]
            summary.append(
                {
                    "method": method,
                    "regime": regime,
                    "threshold": thresholds[method],
                    "mean_window_risk": float(
                        np.mean([float(row[f"{key}_risk"]) for row in selected])
                    ),
                    "alert_rate": float(
                        np.mean([int(row[f"{key}_alert"]) for row in selected])
                    ),
                }
            )
    return rows, summary, dro


def save_outputs(
    outdir: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    stream_windows: list[tuple[str, np.ndarray]],
    dro: DistributionallyRobustPCA,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("this example requires robustcov[plot]") from exc

    outdir.mkdir(parents=True, exist_ok=True)

    windows = np.array([int(row["window"]) for row in rows])
    empirical = np.array([float(row["empirical_normalized"]) for row in rows])
    dro_scores = np.array([float(row["dro_normalized"]) for row in rows])
    boundaries = [0, 12, 24, 36]
    regime_labels = ["Nominal", "Geometry-aligned shift", "Off-geometry drift"]

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.plot(windows, empirical, marker="o", markersize=3.5, label="Empirical PCA")
    ax.plot(windows, dro_scores, marker="s", markersize=3.5, label="DRO-PCA")
    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Calibrated alert threshold")
    for start, stop, label in zip(boundaries[:-1], boundaries[1:], regime_labels):
        ax.axvspan(start + 0.5, stop + 0.5, alpha=0.08)
        ax.text((start + stop + 1) / 2, 1.52, label, ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0.5, len(rows) + 0.5)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("Mean reconstruction risk / calibrated threshold")
    ax.set_title("DRO-PCA tolerates geometry-aligned shift and flags off-geometry drift")
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "drift_timeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    methods = ["Empirical PCA", "DRO-PCA"]
    regimes = ["nominal", "geometry_aligned", "off_geometry"]
    x = np.arange(len(regimes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for offset, method in enumerate(methods):
        values = [
            next(
                float(row["alert_rate"])
                for row in summary
                if row["method"] == method and row["regime"] == regime
            )
            for regime in regimes
        ]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=method)
    ax.set_xticks(x, ["Nominal", "Geometry-aligned", "Off-geometry"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of windows alerted")
    ax.set_title("Alert behavior by drift regime")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "alert_rates.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    aligned_window = next(window for regime, window in stream_windows if regime == "geometry_aligned")
    off_window = next(window for regime, window in stream_windows if regime == "off_geometry")

    def feature_contributions(X: np.ndarray) -> np.ndarray:
        scores = dro.transform(X)
        residuals = X - dro.inverse_transform(scores)
        return np.mean(np.square(residuals), axis=0)

    aligned_contribution = feature_contributions(aligned_window)
    off_contribution = feature_contributions(off_window)
    feature_index = np.arange(1, dro.n_features_in_ + 1)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(feature_index, aligned_contribution, marker="o", label="Geometry-aligned shift")
    ax.plot(feature_index, off_contribution, marker="s", label="Off-geometry drift")
    ax.set_xticks(feature_index)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean squared residual contribution")
    ax.set_title("Residual contributions localize the unmodeled drift directions")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "feature_contributions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    with (outdir / "window_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/use_cases/distributionally_robust_pca_drift_monitoring"),
    )
    parser.add_argument("--false-alarm-rate", type=float, default=0.05)
    args = parser.parse_args()

    X_reference, calibration, stream, _ = build_monitoring_problem()
    rows, summary, dro = run_monitor(
        X_reference,
        calibration,
        stream,
        false_alarm_rate=args.false_alarm_rate,
    )
    save_outputs(args.outdir, rows, summary, stream, dro)

    print("method,regime,threshold,mean_window_risk,alert_rate")
    for row in summary:
        print(
            f"{row['method']},{row['regime']},{float(row['threshold']):.6f},"
            f"{float(row['mean_window_risk']):.6f},{float(row['alert_rate']):.3f}"
        )
    print(f"dro_exact_worst_case_risk,{dro.exact_worst_case_risk_:.6f}")
    print("monitoring_threshold,empirically_calibrated_window_quantile")
    print("worst_case_risk_role,model_diagnostic_not_alert_threshold")
    print(f"saved,{args.outdir}")


if __name__ == "__main__":
    main()
