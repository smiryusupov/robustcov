from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "compare_methods.py"
SPEC = importlib.util.spec_from_file_location("method_comparison_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def test_binary_auc_is_one_for_perfect_ranking():
    labels = np.array([False, True, False, True])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    assert BENCH.binary_auc(labels, scores) == 1.0


def test_projection_error_is_zero_for_same_subspace_with_rotation():
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = rotation @ basis
    assert BENCH.projection_error(rotated, basis) < 1e-12


def test_graph_metrics_count_upper_triangle_once():
    truth = np.array(
        [[False, True, False], [True, False, True], [False, True, False]]
    )
    predicted = np.array(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    precision, recall, f1, n_edges = BENCH.graph_metrics(predicted, truth)
    assert n_edges == 2
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_matrix_family_quick_run_writes_csv(tmp_path):
    output = tmp_path / "matrix.csv"
    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--profile",
            "quick",
            "--families",
            "matrix",
            "--csv",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    text = output.read_text()
    assert "All-sample matrix-normal MLE" in text
    assert "MMCD" in text


def test_high_dimensional_mixed_factories_include_cellrcov():
    factories = BENCH.scatter_factories(
        "high-dimensional mixed contamination", BENCH.PROFILES["quick"]
    )
    names = [name for name, _, _ in factories]
    assert "CellRCov" in names
    assert "CellMCD" not in names
    assert "FastMCD" not in names
