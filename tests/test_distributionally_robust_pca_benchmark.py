from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import robustcov as rc
import robustcov.experimental as experimental


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "distributionally_robust_pca.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("dro_pca_benchmark_test", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def test_experimental_namespace_is_deliberate():
    assert hasattr(experimental, "DistributionallyRobustPCA")
    assert experimental.WassersteinRobustPCA is experimental.DistributionallyRobustPCA
    assert not hasattr(rc, "DistributionallyRobustPCA")


def test_shift_benchmark_distinguishes_distribution_shift_from_contamination_robustness():
    rows = BENCHMARK.run("quick", seed=20260719, repeats=1)
    structured = {
        str(row["method"]): row
        for row in rows
        if row["scenario"] == "structured covariance shift"
    }
    empirical = float(structured["Empirical PCA"]["target_risk"])
    identity = float(structured["DRO-PCA identity control"]["target_risk"])
    residual = float(structured["DRO-PCA residual geometry"]["target_risk"])
    np.testing.assert_allclose(identity, empirical, rtol=2e-12, atol=2e-12)
    assert residual < 0.85 * empirical
    assert float(structured["DRO-PCA residual geometry"]["selected_gamma"]) > 0.0

    contamination = {
        str(row["method"]): row
        for row in rows
        if row["scenario"] == "row contamination without target shift"
    }
    assert float(contamination["RobustPCA(Cauchy)"]["target_risk"]) < float(
        contamination["Empirical PCA"]["target_risk"]
    )


def test_distribution_shift_benchmark_cli_writes_csv_and_plot(tmp_path):
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "dro.csv"
    plot_path = tmp_path / "dro.png"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    script_args = [
        str(BENCHMARK_PATH),
        "--profile",
        "quick",
        "--repeats",
        "1",
        "--csv",
        str(csv_path),
        "--plot",
        str(plot_path),
    ]
    import_paths = [str(ROOT), *[path for path in sys.path if path and "site-packages" in path]]
    bootstrap = (
        "import runpy, sys; "
        f"sys.path[:0] = {import_paths!r}; "
        f"sys.argv = {script_args!r}; "
        f"runpy.run_path({str(BENCHMARK_PATH)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", bootstrap],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert plot_path.is_file() and plot_path.stat().st_size > 10_000
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert {row["scenario"] for row in rows} == {
        "structured covariance shift",
        "no distribution shift",
        "row contamination without target shift",
    }
