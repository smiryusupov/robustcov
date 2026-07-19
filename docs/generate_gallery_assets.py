"""Generate static Sphinx gallery assets from runnable use-case examples.

This script runs each gallery example, captures stdout/stderr, and copies the
plots saved under results/use_cases into docs/_static/gallery/<slug>/ so the
Sphinx pages show real outputs instead of only instructions.

Run from the repository root:
    python docs/generate_gallery_assets.py

Generate selected pages only:
    python docs/generate_gallery_assets.py --only robust_pca_yield_curve
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GalleryCase:
    slug: str
    script: str
    result_dir: str | None
    images: tuple[str, ...]


CASES = [
    GalleryCase("finance_risk", "use_case_finance_risk.py", "finance", ("distance_panel.png", "covariance.png")),
    GalleryCase("portfolio_stress", "use_case_portfolio_stress.py", "portfolio_stress", ("distance_panel.png", "covariance.png")),
    GalleryCase("fraud_screening", "use_case_fraud_screening.py", "fraud", ("distance_panel.png", "distance_profile.png")),
    GalleryCase("sensor_anomaly", "use_case_sensor_anomaly.py", "sensor", ("distance_panel.png", "distance_profile.png")),
    GalleryCase("maintenance_monitoring", "use_case_maintenance_monitoring.py", "maintenance", ("distance_panel.png", "time_profile.png")),
    GalleryCase("quality_control", "use_case_quality_control.py", "quality_control", ("support_ellipse.png", "distance_profile.png")),
    GalleryCase("network_traffic", "use_case_network_traffic.py", "network", ("distance_panel.png",)),
    GalleryCase("biomedical_signal", "use_case_biomedical_signal.py", "biomedical_signal", ("distance_profile.png",)),
    GalleryCase("image_feature_anomaly", "use_case_image_feature_anomaly.py", "image_features", ("distance_panel.png",)),
    GalleryCase("text_embedding_outliers", "use_case_text_embedding_outliers.py", "embedding_outliers", ("distance_panel.png",)),
    GalleryCase("breast_cancer_screening", "use_case_breast_cancer_screening.py", "breast_cancer", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("digits_one_class", "use_case_digits_one_class_baselines.py", "digits_one_class", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("wine_class_screening", "use_case_wine_class_screening.py", "wine_class", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("ml_preprocessing", "use_case_ml_preprocessing.py", "ml_preprocessing", ("accuracy_comparison.png", "distance_profile.png")),
    GalleryCase("gp_robust_input_metric", "gp_robust_input_metric.py", "gp_robust_input_metric", ("kernel_comparison.png",)),
    GalleryCase("mrcd_high_dimensional_outliers", "plot_mrcd_high_dimensional_outliers.py", "mrcd_high_dimensional_outliers", ("distance_comparison.png", "covariance_spectrum.png", "distance_crossplot.png", "metrics.csv")),
    GalleryCase("kmrcd_nonlinear_manifold", "plot_kmrcd_nonlinear_manifold.py", "kmrcd_nonlinear_manifold", ("linear_distance_contours.png", "kernel_distance_contours.png", "auc_comparison.png", "bandwidth_sensitivity.png", "metrics.csv")),
    GalleryCase("dets_detmm_tradeoff", "plot_dets_detmm_tradeoff.py", "dets_detmm_tradeoff", ("robust_ellipses.png", "covariance_error.png", "weight_functions.png", "clean_efficiency.png", "metrics.csv")),
    GalleryCase("mmcd_sensor_windows", "plot_mmcd_sensor_windows.py", "mmcd_sensor_windows", ("distance_comparison.png", "contribution_heatmap.png", "covariance_factors.png", "metrics.csv")),
    GalleryCase("cellmcd_market_data", "plot_cellmcd_market_data.py", "cellmcd_market_data", ("covariance_error.png", "cell_residual_map.png", "correlation_comparison.png", "metrics.csv")),
    GalleryCase("cellpca_process_spectra", "plot_cellpca_process_spectra.py", "cellpca_process_spectra", ("subspace_recovery.png", "residual_cellmap.png", "outlier_map.png", "loading_curves.png", "metrics.csv")),
    GalleryCase("sparse_cellpca_spectra", "plot_sparse_cellpca_spectra.py", "sparse_cellpca_spectra", ("loading_comparison.png", "performance_comparison.png", "sparse_loadings.png", "outlier_map.png", "metrics.csv")),
    GalleryCase("cellrcov_high_dimensional", "plot_cellrcov_high_dimensional.py", "cellrcov_high_dimensional", ("covariance_error.png", "covariance_spectrum.png", "distance_decomposition.png", "cell_residual_map.png", "metrics.csv")),
    GalleryCase("robust_graphical_lasso_market_network", "plot_robust_graphical_lasso_market_network.py", "robust_graphical_lasso_market_network", ("partial_correlation_comparison.png", "robust_network.png", "ebic_path.png", "metrics.csv")),
    GalleryCase("spatial_sign_graphical_lasso", "plot_spatial_sign_graphical_lasso.py", "spatial_sign_graphical_lasso", ("partial_correlation_comparison.png", "spatial_sign_network.png", "graph_recovery.png", "radial_stability.png", "metrics.csv")),
    GalleryCase("robust_pca_embedding_monitoring", "plot_robust_pca_embedding_monitoring.py", "robust_pca_embedding_monitoring", ("batch_monitoring.png", "outlier_map.png", "subspace_recovery.png")),
    GalleryCase("density_power_pca", "plot_density_power_pca.py", "density_power_pca", ("subspace_comparison.png", "alpha_tradeoff.png", "cell_weight_map.png", "outlier_map.png", "metrics.csv")),
    GalleryCase("robust_subspace_monitoring", "plot_robust_subspace_monitoring.py", "robust_subspace_monitoring", ("monitor_history.png", "drift_mechanism_map.png", "final_batch_outlier_map.png")),
    GalleryCase("robust_pca_yield_curve", "plot_robust_pca_yield_curve.py", "robust_pca_yield_curve", ("factor_loadings.png", "factor_scores.png", "outlier_map.png", "metrics.csv")),
    GalleryCase("robust_pca_subspace_stability", "plot_robust_pca_subspace_stability.py", "robust_pca_subspace_stability", ("loading_intervals.png", "principal_angle_distribution.png", "eigenvalue_intervals.png", "metrics.csv")),
    GalleryCase("robust_pca_dependent_stability", "plot_robust_pca_dependent_stability.py", "robust_pca_dependent_stability", ("loading_intervals.png", "principal_angle_distribution.png", "eigenvalue_uncertainty.png", "metrics.csv")),
    GalleryCase("robust_pca_market_risk", "plot_robust_pca_market_risk.py", "robust_pca_market_risk", ("asset_loadings.png", "explained_variance.png", "outlier_map.png", "reconstruction_residual.png")),
    GalleryCase("multimodal_anomaly", "use_case_multimodal_anomaly.py", "multimodal_anomaly", ("cluster_distance_panel.png", "global_distance_profile.png", "metrics.csv")),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="SLUG",
        help="Generate only the selected gallery slugs.",
    )
    args = parser.parse_args()

    cases = CASES
    if args.only:
        requested = set(args.only)
        known = {case.slug for case in CASES}
        unknown = sorted(requested - known)
        if unknown:
            parser.error("unknown gallery slug(s): " + ", ".join(unknown))
        cases = [case for case in CASES if case.slug in requested]

    root = Path(__file__).resolve().parents[1]
    out_root = root / "docs" / "_static" / "gallery"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int]] = []

    for case in cases:
        script_path = root / "examples" / case.script
        case_out = out_root / case.slug
        case_out.mkdir(parents=True, exist_ok=True)
        print(f"running {case.script}")
        try:
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=45,
                env=env,
            )
            returncode = proc.returncode
            output = proc.stdout
            if proc.stderr:
                output += "\n[stderr]\n" + proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            output = (exc.stdout or "")
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            err = exc.stderr or ""
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            output += "\n[timeout] example exceeded 45 seconds\n" + err
        (case_out / "output.txt").write_text(output.strip() + "\n", encoding="utf-8")

        copied = []
        if case.result_dir is not None:
            source_dir = root / "results" / "use_cases" / case.result_dir
            for image in case.images:
                src = source_dir / image
                if src.exists():
                    shutil.copy2(src, case_out / image)
                    copied.append(image)
        (case_out / "manifest.txt").write_text("\n".join(copied) + ("\n" if copied else ""), encoding="utf-8")
        rows.append((case.script, returncode))

    summary_path = out_root / "summary.csv"
    statuses: dict[str, str] = {}
    if args.only and summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines()[1:]:
            if "," in line:
                script, status = line.split(",", 1)
                statuses[script] = status
    for script, code in rows:
        statuses[script] = "ok" if code == 0 else f"failed({code})"

    ordered_scripts = [case.script for case in CASES if case.script in statuses]
    ordered_scripts.extend(
        script for script in statuses if script not in set(ordered_scripts)
    )
    summary = ["script,status"] + [
        f"{script},{statuses[script]}" for script in ordered_scripts
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(["script,status"] + [
        f"{script},{'ok' if code == 0 else f'failed({code})'}"
        for script, code in rows
    ]))
    return 1 if any(code != 0 for _, code in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
