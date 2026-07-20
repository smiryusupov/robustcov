"""Run robustcov examples by method family or application group.

Examples
--------
Run the compact default set::

    python examples/run_use_case_gallery.py

Run a method family::

    python examples/run_use_case_gallery.py --group ica
    python examples/run_use_case_gallery.py --group pca
    python examples/run_use_case_gallery.py --group robust
    python examples/run_use_case_gallery.py --group monitoring

List groups or run every registered gallery script::

    python examples/run_use_case_gallery.py --list
    python examples/run_use_case_gallery.py --all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GROUPS: dict[str, list[str]] = {
    "quick": [
        "ica_two_scatter.py",
        "robust_factor_model.py",
        "use_case_finance_risk.py",
        "use_case_fraud_screening.py",
        "use_case_sensor_anomaly.py",
        "use_case_breast_cancer_screening.py",
    ],
    "ica": [
        "ica_two_scatter.py",
        "sobi_source_separation.py",
        "source_separation_and_factor_models.py",
    ],
    "pca": [
        "principal_component_pursuit.py",
        "distributionally_robust_pca.py",
        "distributionally_robust_pca_drift_monitoring.py",
        "online_robust_subspace_tracking.py",
        "robust_factor_model.py",
        "plot_robust_pca_yield_curve.py",
        "plot_robust_pca_subspace_stability.py",
        "plot_robust_pca_market_risk.py",
        "plot_cellpca_process_spectra.py",
        "plot_sparse_cellpca_spectra.py",
        "plot_density_power_pca.py",
        "plot_robust_multilinear_pca.py",
    ],
    "robust": [
        "adversarial_covariance_filtering.py",
        "use_case_finance_risk.py",
        "use_case_portfolio_stress.py",
        "plot_mrcd_high_dimensional_outliers.py",
        "plot_kmrcd_nonlinear_manifold.py",
        "plot_dets_detmm_tradeoff.py",
        "plot_cellmcd_market_data.py",
        "plot_cellrcov_high_dimensional.py",
        "plot_mmcd_sensor_windows.py",
        "plot_robust_graphical_lasso_market_network.py",
        "plot_spatial_sign_graphical_lasso.py",
    ],
    "monitoring": [
        "use_case_fraud_screening.py",
        "use_case_network_traffic.py",
        "use_case_sensor_anomaly.py",
        "use_case_maintenance_monitoring.py",
        "use_case_quality_control.py",
        "plot_robust_subspace_monitoring.py",
        "distributionally_robust_pca_drift_monitoring.py",
        "online_robust_subspace_tracking.py",
        "feature_geometry_drift_detection.py",
        "feature_geometry_embedding_monitoring.py",
        "use_case_breast_cancer_screening.py",
        "use_case_digits_one_class_baselines.py",
        "use_case_wine_class_screening.py",
        "use_case_ml_preprocessing.py",
    ],
}


def all_scripts() -> list[str]:
    """Return registered scripts once, preserving group order."""

    return list(dict.fromkeys(script for scripts in GROUPS.values() for script in scripts))


def run_script(path: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print(f"\n$ {sys.executable} {path}")
    try:
        return subprocess.run(
            [sys.executable, str(path)],
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        err += f"\n[timeout] example exceeded {timeout} seconds"
        return subprocess.CompletedProcess([sys.executable, str(path)], 124, out, err)


def print_groups() -> None:
    print("registered example groups")
    for name, scripts in GROUPS.items():
        print(f"\n{name}")
        for script in scripts:
            print(f"  {script}")


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="run every registered gallery example")
    selection.add_argument("--group", choices=tuple(GROUPS), help="run one method/application group")
    selection.add_argument("--list", action="store_true", help="list groups and their scripts")
    parser.add_argument("--timeout", type=int, default=60, help="per-example timeout in seconds")
    args = parser.parse_args()

    if args.list:
        print_groups()
        return

    group_name = args.group or "quick"
    scripts = all_scripts() if args.all else GROUPS[group_name]
    here = Path(__file__).resolve().parent

    missing = [script for script in scripts if not (here / script).is_file()]
    if missing:
        raise FileNotFoundError(f"registered example scripts are missing: {missing}")

    rows: list[tuple[str, int]] = []
    for script in scripts:
        proc = run_script(here / script, args.timeout)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())
        rows.append((script, proc.returncode))

    label = "all" if args.all else group_name
    print(f"\nuse-case gallery summary ({label})")
    for script, code in rows:
        status = "ok" if code == 0 else f"failed({code})"
        print(f"{script},{status}")
    if any(code != 0 for _, code in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
