from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CATALOG = _load(ROOT / "benchmarks" / "covariance_catalog.py", "covariance_catalog_test")
SUMMARY = _load(ROOT / "benchmarks" / "benchmark_summary.py", "benchmark_summary_scope_test")
OPENMP = _load(ROOT / "benchmarks" / "openmp_scaling.py", "openmp_scope_test")


def test_covariance_catalog_covers_current_relevant_estimators():
    names = set(CATALOG.method_names(purpose="accuracy"))
    expected = {
        "robustcov FastMCD",
        "robustcov MRCD",
        "robustcov DetS",
        "robustcov DetMM",
        "robustcov TylerShape",
        "robustcov RegularizedTyler",
        "robustcov KLRegularizedTyler",
        "robustcov WieselTyler",
        "robustcov StudentT(df=3)",
        "robustcov RegularizedCauchy",
        "robustcov HellingerTyler (experimental)",
        "robustcov AutoRobustScatter",
    }
    assert expected <= names


def test_catalog_marks_structurally_inapplicable_methods_in_high_dimension():
    methods = {method.name: method for method in CATALOG.covariance_methods()}
    for name in (
        "robustcov FastMCD",
        "robustcov DetS",
        "robustcov DetMM",
        "robustcov TylerShape",
        "sklearn MinCovDet",
    ):
        if name in methods:
            applicable, reason = methods[name].applicable(40, 80)
            assert applicable is False
            assert reason
    assert methods["robustcov MRCD"].applicable(40, 80)[0] is True
    assert methods["robustcov RegularizedCauchy"].applicable(40, 80)[0] is True


def test_summary_excludes_not_applicable_rows_from_failure_rate():
    rows = [
        {"scenario": "a", "method": "A", "status": "ok", "error": "1.0", "seconds": "0.1"},
        {"scenario": "b", "method": "A", "status": "not_applicable", "error": "", "seconds": ""},
        {"scenario": "a", "method": "B", "status": "failed", "error": "", "seconds": ""},
        {"scenario": "b", "method": "B", "status": "ok", "error": "2.0", "seconds": "0.2"},
    ]
    summary = {
        row["method"]: row
        for row in SUMMARY.summarize(rows, "method", "error", "seconds", ["scenario"])
    }
    assert summary["A"]["eligible"] == 1
    assert summary["A"]["not_applicable"] == 1
    assert summary["A"]["failures"] == 0
    assert summary["A"]["success_rate"] == "1.0000"
    assert summary["B"]["eligible"] == 2
    assert summary["B"]["failures"] == 1
    assert summary["B"]["success_rate"] == "0.5000"


def test_speed_cli_keeps_inapplicable_rows_visible(tmp_path):
    output = tmp_path / "speed.csv"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "speed_estimators.py"),
            "--profile", "quick",
            "--workloads", "classical_contamination", "high_dimensional",
            "--repeat", "1",
            "--exclude-experimental",
            "--exclude-selector",
            "--methods", "robustcov FastMCD", "robustcov MRCD",
            "--csv", str(output),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["workload"], row["method"]): row for row in rows}
    assert indexed[("classical_contamination", "robustcov FastMCD")]["status"] == "ok"
    assert indexed[("high_dimensional", "robustcov FastMCD")]["status"] == "not_applicable"
    assert indexed[("high_dimensional", "robustcov MRCD")]["status"] == "ok"


def test_openmp_catalog_lists_every_threaded_native_region():
    assert set(OPENMP.OPENMP_WORKLOAD_KEYS) == {
        "fastmcd",
        "tyler",
        "regularized_tyler",
        "vector_mahalanobis",
        "matrix_mahalanobis",
        "weighted_tucker",
    }
