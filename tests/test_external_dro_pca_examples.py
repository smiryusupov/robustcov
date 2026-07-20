# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from examples_external.cmapss_dro_pca_monitoring import run_analysis as run_cmapss
from examples_external.gas_sensor_drift_dro_pca import run_analysis as run_gas


def test_gas_sensor_external_analysis_runs_on_synthetic_domains():
    rng = np.random.default_rng(0)
    rows = []
    gases = []
    concentrations = []
    batches = []
    p = 40
    gas_effect = rng.normal(scale=0.7, size=(6, p))
    concentration_effect = rng.normal(scale=0.1, size=(6, p))
    for batch in range(1, 11):
        drift = np.zeros(p)
        drift[4:8] = 0.12 * batch
        for gas in range(1, 7):
            for repetition in range(8):
                concentration = 10.0 + 5.0 * repetition
                value = (
                    gas_effect[gas - 1]
                    + np.log1p(concentration) * concentration_effect[gas - 1]
                    + drift
                    + rng.normal(scale=0.5, size=p)
                )
                rows.append(value)
                gases.append(gas)
                concentrations.append(concentration)
                batches.append(batch)
    window_rows, summary, metadata = run_gas(
        np.asarray(rows),
        np.asarray(gases),
        np.asarray(concentrations),
        np.asarray(batches),
        n_components=4,
        window_size=16,
        window_step=8,
    )
    assert window_rows
    assert len(summary) == 22
    assert metadata["transport_geometry"] == "batch_mean_shift_diagonal"
    failure = [row for row in summary if row["scenario"] == "synthetic_off_geometry_sensor_failure"]
    assert {row["method"] for row in failure} == {"Empirical PCA", "DRO-PCA"}


def test_cmapss_external_analysis_runs_on_synthetic_run_to_failure_data():
    rng = np.random.default_rng(1)
    sensors = []
    settings = []
    units = []
    cycles = []
    p = 12
    regime_effects = rng.normal(scale=0.8, size=(4, p))
    degradation_direction = np.zeros(p)
    degradation_direction[-3:] = [1.0, -0.8, 0.6]
    for unit in range(1, 9):
        for cycle in range(1, 101):
            regime = (cycle // 10 + unit) % 4
            life = cycle / 100.0
            degradation = max(life - 0.45, 0.0) ** 2 * 8.0
            sensors.append(
                regime_effects[regime]
                + degradation * degradation_direction
                + rng.normal(scale=0.35, size=p)
            )
            settings.append([regime, regime % 2, (regime // 2) % 2])
            units.append(unit)
            cycles.append(cycle)
    rows, summary, metadata, contributions = run_cmapss(
        np.asarray(sensors),
        np.asarray(settings, dtype=float),
        np.asarray(units),
        np.asarray(cycles),
        n_components=4,
        window_size=10,
        step=5,
    )
    assert rows
    assert summary
    assert contributions.shape == (p,)
    assert metadata["n_operating_regimes"] >= 2
    assert metadata["threshold"] == "split_conformal_upper_tail_p_value"
    assert all(0.0 < float(row["conformal_p_value"]) <= 1.0 for row in rows)
    late_dro = next(
        row for row in summary if row["method"] == "DRO-PCA" and row["life_bin"] == "0.8-1.0"
    )
    early_dro = next(
        row for row in summary if row["method"] == "DRO-PCA" and row["life_bin"] == "0.0-0.2"
    )
    assert float(late_dro["mean_risk"]) > float(early_dro["mean_risk"])
