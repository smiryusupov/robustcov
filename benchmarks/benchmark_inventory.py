#!/usr/bin/env python3
"""Audit benchmark coverage across the public robustcov estimator surface.

The inventory distinguishes comparative benchmarks from numerical validation,
performance gates, and workflow-only coverage.  It is intentionally explicit:
adding a new public estimator should add or update one row here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import robustcov as rc
import robustcov.experimental as experimental


@dataclass(frozen=True)
class Coverage:
    family: str
    estimator: str
    level: str
    benchmark: str
    primary_metric: str
    notes: str = ""


COVERAGE = [
    Coverage("scatter", "FastMCD", "comparative", "benchmarks/compare_methods.py", "covariance error / row AUROC"),
    Coverage("scatter", "MRCD", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "KMRCD", "comparative", "benchmarks/compare_methods.py", "outlier AUROC"),
    Coverage("scatter", "DetS", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "DetMM", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "TylerShape", "validation", "benchmarks/statistical_validation.py", "scale/equivariance checks"),
    Coverage("scatter", "RegularizedTyler", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "StudentTScatter", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "RegularizedCauchy", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("scatter", "KLRegularizedTyler", "comparative", "benchmarks/small_sample_heavy_tail.py", "covariance error"),
    Coverage("scatter", "WieselTyler", "validation", "benchmarks/statistical_validation.py", "shape/equivariance checks"),
    Coverage("scatter", "HellingerRegularizedTyler", "comparative", "benchmarks/small_sample_heavy_tail.py", "covariance error", "experimental approximation"),
    Coverage("cellwise scatter", "CellMCD", "comparative", "benchmarks/compare_methods.py", "cell AUROC / covariance error"),
    Coverage("cellwise scatter", "CellRCov", "comparative", "benchmarks/compare_methods.py", "covariance error"),
    Coverage("matrix scatter", "MatrixMCD", "comparative", "benchmarks/compare_methods.py", "Kronecker covariance error"),
    Coverage("matrix decomposition", "PrincipalComponentPursuit", "comparative", "benchmarks/principal_component_pursuit_validation.py", "low-rank recovery error / sparse-support recovery"),
    Coverage("PCA", "RobustPCA", "comparative", "benchmarks/latent_structure_benchmarks.py", "subspace error / row AUROC"),
    Coverage("PCA", "DensityPowerRobustPCA", "comparative", "benchmarks/latent_structure_benchmarks.py", "subspace error"),
    Coverage("PCA", "CellPCA", "comparative", "benchmarks/latent_structure_benchmarks.py", "cell AUROC / missing-value MAE"),
    Coverage("PCA", "SparseCellPCA", "comparative", "benchmarks/compare_methods.py", "support F1 / subspace error"),
    Coverage("PCA", "RobustMultilinearPCA", "comparative", "benchmarks/compare_methods.py", "mode-subspace error / reconstruction MAE"),
    Coverage("source separation", "TwoScatterICA", "comparative", "benchmarks/latent_structure_benchmarks.py", "minimum-distance index"),
    Coverage("source separation", "SOBI", "comparative", "benchmarks/latent_structure_benchmarks.py", "minimum-distance index"),
    Coverage("source separation", "RobustSOBI", "comparative", "benchmarks/latent_structure_benchmarks.py", "minimum-distance index under impulses"),
    Coverage("factor models", "RobustFactorModel", "comparative", "benchmarks/latent_structure_benchmarks.py", "loading-subspace/common-component error"),
    Coverage("precision", "RobustGraphicalLasso", "comparative", "benchmarks/compare_methods.py", "edge F1 / partial-correlation error"),
    Coverage("precision", "SGLASSO", "comparative", "benchmarks/compare_methods.py", "edge F1 / partial-correlation error"),
    Coverage("anomaly detection", "RobustOutlierDetector", "comparative", "benchmarks/anomaly_detection_baselines.py", "F1 / ROC AUC"),
    Coverage("anomaly detection", "AutoRobustAnomalyDetector", "comparative", "benchmarks/anomaly_detection_baselines.py", "F1 / ROC AUC"),
    Coverage("anomaly detection", "ClusterRobustOutlierDetector", "workflow", "examples/use_case_multimodal_anomaly.py", "cluster-conditioned anomaly diagnostics"),
    Coverage("selection", "AutoRobustScatter", "comparative", "benchmarks/auto_scatter_small_sample.py", "selection score / covariance error"),
    Coverage("monitoring", "RobustSubspaceMonitor", "validation", "benchmarks/precision_geometry_monitoring_validation.py", "drift invariance / calibration"),
    Coverage("monitoring", "ConformalAlertCalibrator", "validation", "benchmarks/conformal_alert_calibration_validation.py", "marginal false-alert rate / contaminated-reference conservativeness"),
    Coverage("monitoring", "SubspaceStability", "workflow", "examples/plot_robust_pca_dependent_stability.py", "bootstrap subspace stability"),
    Coverage("geometry", "FeatureGeometry", "validation", "benchmarks/precision_geometry_monitoring_validation.py", "distance invariance"),
    Coverage("geometry", "ClassConditionalFeatureGeometry", "workflow", "examples/feature_geometry_class_conditional_ood.py", "class-conditional OOD ranking"),
    Coverage("geometry", "RobustInputMetric", "workflow", "examples/gp_robust_input_metric.py", "kernel/input-metric behavior"),
    Coverage("preprocessing", "RobustMedianImputer", "workflow", "examples/use_case_ml_preprocessing.py", "downstream classification"),
    Coverage("scatter base", "IterativeMScatter", "performance", "benchmarks/estimator_optimization_gate.py", "complete-fit runtime/equivalence"),
]

EXPERIMENTAL_COVERAGE = [
    Coverage(
        "experimental scatter",
        "SpectralFilteringCovariance",
        "validation",
        "benchmarks/spectral_filter_covariance_validation.py",
        "relative covariance error / adversarial-row recall",
        "practical robustcov spectral-filtering composite; not the optimal Gaussian filtering algorithm",
    ),
    Coverage(
        "experimental monitoring",
        "OnlineRobustSubspaceTracker",
        "validation",
        "benchmarks/online_subspace_tracking_validation.py",
        "projector error under gradual rotation / corruption-screening diagnostics",
        "robustcov composite inspired by online robust subspace-tracking research; not NORST",
    ),
    Coverage(
        "experimental PCA",
        "DistributionallyRobustPCA",
        "comparative",
        "benchmarks/distributionally_robust_pca.py",
        "held-out target reconstruction risk under distribution shift",
        "exact weighted-Wasserstein risk evaluated over a deterministic candidate path",
    ),
]


# Public aliases intentionally share their canonical estimator's benchmark.
ALIASES = {
    "MinCovDet": "FastMCD",
    "MinimumRegularizedCovarianceDeterminant": "MRCD",
    "MinRegularizedCovDet": "MRCD",
    "KernelMRCD": "KMRCD",
    "KernelMinimumRegularizedCovarianceDeterminant": "KMRCD",
    "DeterministicSEstimator": "DetS",
    "DeterministicMMEstimator": "DetMM",
    "MMCD": "MatrixMCD",
    "MatrixMinimumCovarianceDeterminant": "MatrixMCD",
    "CellwiseMCD": "CellMCD",
    "CellwiseMinimumCovarianceDeterminant": "CellMCD",
    "CellwiseRegularizedCovariance": "CellRCov",
    "CellwiseRobustCovariance": "CellRCov",
    "PCP": "PrincipalComponentPursuit",
    "DPDRobustPCA": "DensityPowerRobustPCA",
    "CellwiseRobustPCA": "CellPCA",
    "CasewiseCellwisePCA": "CellPCA",
    "SparseCellwiseRobustPCA": "SparseCellPCA",
    "SparseCasewiseCellwisePCA": "SparseCellPCA",
    "CasewiseCellwiseMultilinearPCA": "RobustMultilinearPCA",
    "CellwiseRobustMultilinearPCA": "RobustMultilinearPCA",
    "SparseRobustPrecision": "RobustGraphicalLasso",
    "SpatialSignGraphicalLasso": "SGLASSO",
    "SpatialSignSparsePrecision": "SGLASSO",
}

EXPERIMENTAL_ALIASES = {
    "WassersteinRobustPCA": "DistributionallyRobustPCA",
}


def validate(root: Path) -> list[str]:
    errors = []
    covered = {entry.estimator for entry in COVERAGE}
    experimental_covered = {entry.estimator for entry in EXPERIMENTAL_COVERAGE}
    for entry in COVERAGE:
        if not hasattr(rc, entry.estimator):
            errors.append(f"public estimator is missing: {entry.estimator}")
        if not (root / entry.benchmark).is_file():
            errors.append(f"benchmark path is missing: {entry.benchmark}")
    for entry in EXPERIMENTAL_COVERAGE:
        if not hasattr(experimental, entry.estimator):
            errors.append(f"experimental estimator is missing: {entry.estimator}")
        if not (root / entry.benchmark).is_file():
            errors.append(f"benchmark path is missing: {entry.benchmark}")
    for alias, canonical in ALIASES.items():
        if not hasattr(rc, alias):
            errors.append(f"public alias is missing: {alias}")
        if canonical not in covered:
            errors.append(f"alias {alias} references uncovered canonical estimator {canonical}")
    for alias, canonical in EXPERIMENTAL_ALIASES.items():
        if not hasattr(experimental, alias):
            errors.append(f"experimental alias is missing: {alias}")
        if canonical not in experimental_covered:
            errors.append(
                f"experimental alias {alias} references uncovered estimator {canonical}"
            )
    return errors


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(entry) for entry in (*COVERAGE, *EXPERIMENTAL_COVERAGE)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_rst(path: Path) -> None:
    lines = [
        ".. Generated by benchmarks/benchmark_inventory.py.",
        "",
        "Benchmark coverage inventory",
        "----------------------------",
        "",
        "``comparative`` means the estimator appears in a task-specific accuracy/timing comparison. ``validation`` means it has numerical/statistical gates, while ``workflow`` means it is exercised through an end-to-end use case.",
        "",
        ".. benchmark-inventory-body-start",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Family",
        "     - Estimator",
        "     - Coverage",
        "     - Primary metric",
        "     - Benchmark",
    ]
    for entry in (*COVERAGE, *EXPERIMENTAL_COVERAGE):
        lines.extend([
            f"   * - {entry.family}",
            f"     - ``{entry.estimator}``",
            f"     - {entry.level}",
            f"     - {entry.primary_metric}",
            f"     - ``{entry.benchmark}``",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--rst", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = validate(root)
    all_coverage = (*COVERAGE, *EXPERIMENTAL_COVERAGE)
    comparative = sum(entry.level == "comparative" for entry in all_coverage)
    print(f"canonical estimators inventoried: {len(COVERAGE)}")
    print(f"experimental estimators inventoried: {len(EXPERIMENTAL_COVERAGE)}")
    print(f"comparative benchmark coverage: {comparative}")
    print(f"public aliases mapped: {len(ALIASES)}")
    print(f"experimental aliases mapped: {len(EXPERIMENTAL_ALIASES)}")
    for error in errors:
        print(f"ERROR: {error}")
    if args.csv:
        write_csv(args.csv)
    if args.rst:
        write_rst(args.rst)
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
