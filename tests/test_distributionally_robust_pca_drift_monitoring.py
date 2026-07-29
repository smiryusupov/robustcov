"""Regression tests for the DRO-PCA drift-monitoring example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "distributionally_robust_pca_drift_monitoring.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("dro_pca_drift_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dro_monitor_tolerates_aligned_shift_and_flags_off_geometry_drift():
    example = _load_example_module()
    X_reference, calibration, stream, _ = example.build_monitoring_problem()
    _, summary, dro = example.run_monitor(X_reference, calibration, stream)

    results = {
        (row["method"], row["regime"]): float(row["alert_rate"])
        for row in summary
    }
    assert results[("Empirical PCA", "geometry_aligned")] == 1.0
    assert results[("DRO-PCA", "geometry_aligned")] == 0.0
    assert results[("DRO-PCA", "off_geometry")] == 1.0
    assert results[("Empirical PCA", "off_geometry")] == 1.0
    assert results[("DRO-PCA", "nominal")] <= 0.10

    dro_threshold = next(
        float(row["threshold"])
        for row in summary
        if row["method"] == "DRO-PCA" and row["regime"] == "nominal"
    )
    assert dro.exact_worst_case_risk_ > dro_threshold
    assert not np.isclose(dro.exact_worst_case_risk_, dro_threshold)


def test_calibration_quantile_validates_inputs():
    example = _load_example_module()
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert example._upper_calibration_quantile(values, 0.25) == 4.0

    for invalid in (0.0, 1.0, -0.1, 1.1):
        try:
            example._upper_calibration_quantile(values, invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("invalid false_alarm_rate must be rejected")
