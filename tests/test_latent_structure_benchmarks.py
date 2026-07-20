from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "benchmarks" / "compare_methods.py"
INVENTORY_PATH = ROOT / "benchmarks" / "benchmark_inventory.py"
LATENT_PATH = ROOT / "benchmarks" / "latent_structure_benchmarks.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load(COMPARE_PATH, "compare_methods_latent_test")
INVENTORY = _load(INVENTORY_PATH, "benchmark_inventory_test")


def test_matched_component_correlation_handles_permutation_sign_and_scale():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(200, 3))
    estimate = truth[:, [2, 0, 1]] * np.array([-4.0, 0.5, 2.0])
    assert BENCH.matched_component_correlation(estimate, truth) > 1.0 - 1e-12


def test_robust_ica_beats_nonrobust_baselines_under_impulses():
    rows = BENCH.run_ica_benchmarks(BENCH.PROFILES["quick"], repeats=1, seed=20260718)
    contaminated = {
        row["method"]: row for row in rows
        if row["scenario"] == "impulsive row contamination" and row["status"] == "ok"
    }
    robust = float(contaminated["TwoScatterICA"]["minimum_distance_index"])
    classical = float(contaminated["Classical two-scatter ICA"]["minimum_distance_index"])
    assert robust < 0.15
    assert robust < classical
    if "sklearn FastICA" in contaminated:
        assert robust < float(contaminated["sklearn FastICA"]["minimum_distance_index"])


def test_robust_sobi_beats_classical_sobi_under_impulses():
    rows = BENCH.run_sobi_benchmarks(BENCH.PROFILES["quick"], repeats=1, seed=20260718)
    contaminated = {
        row["method"]: row for row in rows
        if row["scenario"] == "impulsive temporal contamination" and row["status"] == "ok"
    }
    robust = float(contaminated["RobustSOBI"]["minimum_distance_index"])
    classical = float(contaminated["SOBI"]["minimum_distance_index"])
    assert robust < 0.10
    assert robust < classical


def test_robust_factor_model_recovers_loading_space_under_row_contamination():
    rows = BENCH.run_factor_benchmarks(BENCH.PROFILES["quick"], repeats=1, seed=20260718)
    contaminated = {
        row["method"]: row for row in rows
        if row["scenario"] == "factor model + row contamination" and row["status"] == "ok"
    }
    robust = float(contaminated["RobustFactorModel(kendall)"]["factor_subspace_error"])
    empirical = float(contaminated["Empirical PCA factors"]["factor_subspace_error"])
    assert robust < 0.10
    assert robust < empirical
    assert int(contaminated["RobustFactorModel(auto)"]["factor_count_error"]) == 0


def test_latent_benchmark_cli_writes_csv_plots_and_rst(tmp_path):
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "latent.csv"
    plot_dir = tmp_path / "plots"
    rst_path = tmp_path / "latent.rst"
    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(LATENT_PATH),
            "--profile", "quick",
            "--families", "ica", "sobi", "factor",
            "--csv", str(csv_path),
            "--plot-dir", str(plot_dir),
            "--rst", str(rst_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["family"] for row in rows} == {"ica", "sobi", "factor model"}
    assert {path.name for path in plot_dir.glob("*.png")} == {
        "ica_mdi.png", "sobi_mdi.png", "factor_subspace.png"
    }
    assert "Latent-structure benchmark snapshot" in rst_path.read_text()


def test_benchmark_inventory_is_complete_and_paths_exist():
    errors = INVENTORY.validate(ROOT)
    assert errors == []
    canonical = {entry.estimator for entry in INVENTORY.COVERAGE}
    experimental = {entry.estimator for entry in INVENTORY.EXPERIMENTAL_COVERAGE}
    assert {"TwoScatterICA", "RobustSOBI", "RobustPCA", "RobustFactorModel"} <= canonical
    assert experimental == {"DistributionallyRobustPCA"}
