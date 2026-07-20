#!/usr/bin/env python3
"""Deterministic recovery checks for ICA, SOBI, and robust factor models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import robustcov as rc


def run_validation():
    rng = np.random.default_rng(42)
    mixing = np.array([[1.0, 0.4, -0.2], [0.2, 1.2, 0.5], [-0.4, 0.3, 0.9]])
    n = 3000
    independent = np.column_stack([
        rng.laplace(size=n) / np.sqrt(2.0),
        rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n),
        rng.normal(size=n),
    ])
    X_ica = independent @ mixing.T
    ica = rc.TwoScatterICA(radial_clip_quantile=0.9).fit(X_ica)

    coefficients = np.array([0.85, -0.55, 0.2])
    temporal = np.zeros((n, 3))
    innovations = rng.normal(size=(n, 3))
    for index in range(1, n):
        temporal[index] = coefficients * temporal[index - 1] + innovations[index]
    X_sobi = temporal @ mixing.T
    rows = rng.choice(n, 50, replace=False)
    contaminated = X_sobi.copy()
    contaminated[rows] += rng.normal(scale=30.0, size=(rows.size, 3))
    classical = rc.SOBI(lags=12).fit(contaminated)
    robust = rc.RobustSOBI(lags=12).fit(contaminated)

    n_samples, n_features, n_factors = 500, 15, 3
    loadings, _ = np.linalg.qr(rng.normal(size=(n_features, n_factors)))
    factors = rng.standard_t(4, size=(n_samples, n_factors))
    factor_data = factors @ loadings.T + 0.25 * rng.normal(size=(n_samples, n_features))
    factor_rows = rng.choice(n_samples, 30, replace=False)
    factor_data[factor_rows] += rng.normal(scale=8.0, size=(factor_rows.size, n_features))
    model = rc.RobustFactorModel(n_factors="auto", max_factors=6).fit(factor_data)
    projection_error = np.linalg.norm(
        model.loadings_ @ model.loadings_.T - loadings @ loadings.T, ord="fro"
    ) / np.sqrt(2.0 * n_factors)

    result = {
        "ica_mdi": rc.minimum_distance_index(ica.unmixing_, mixing),
        "classical_sobi_mdi_contaminated": rc.minimum_distance_index(classical.unmixing_, mixing),
        "robust_sobi_mdi_contaminated": rc.minimum_distance_index(robust.unmixing_, mixing),
        "selected_factors": model.n_factors_,
        "factor_subspace_error": float(projection_error),
    }
    result["passed"] = bool(
        result["ica_mdi"] < 0.15
        and result["robust_sobi_mdi_contaminated"] < 0.08
        and result["robust_sobi_mdi_contaminated"] < result["classical_sobi_mdi_contaminated"]
        and result["selected_factors"] == 3
        and result["factor_subspace_error"] < 0.1
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_validation()
    print(json.dumps(result, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
