#!/usr/bin/env python3
"""Benchmark weighted-Wasserstein distributionally robust PCA.

This benchmark evaluates the method on *held-out target distributions*.  It
separates covariance shift from no-shift efficiency and row contamination, so a
DRO estimator is not mistaken for a generic outlier-resistant PCA method.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import robustcov as rc
from robustcov.experimental import DistributionallyRobustPCA


PROFILES = {
    "quick": dict(n_train=260, n_target=5000, repeats=3),
    "full": dict(n_train=600, n_target=30000, repeats=10),
}


def _basis_from_covariance(covariance: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return vectors[:, np.argsort(values)[::-1][:rank]]


def _empirical_fit(X: np.ndarray, rank: int) -> dict[str, object]:
    location = np.mean(X, axis=0)
    centered = X - location
    covariance = centered.T @ centered / len(X)
    basis = _basis_from_covariance(covariance, rank)
    return {"basis": basis, "location": location, "estimator": None}


def _robust_fit(X: np.ndarray, rank: int) -> dict[str, object]:
    estimator = rc.RobustPCA(
        n_components=rank,
        estimator=rc.RegularizedCauchy(alpha=0.10, max_iter=200, tol=1e-8),
    ).fit(X)
    return {
        "basis": estimator.components_.T,
        "location": estimator.location_,
        "estimator": estimator,
    }


def _dro_fit(X: np.ndarray, rank: int, geometry: str, radius: float) -> dict[str, object]:
    estimator = DistributionallyRobustPCA(
        n_components=rank,
        radius=radius,
        transport_geometry=geometry,
        formulation="exact",
    ).fit(X)
    return {
        "basis": estimator.components_.T,
        "location": estimator.location_,
        "estimator": estimator,
    }


def _risk(X: np.ndarray, location: np.ndarray, basis: np.ndarray) -> float:
    centered = X - location
    residual = centered - (centered @ basis) @ basis.T
    return float(np.mean(np.einsum("ij,ij->i", residual, residual)))


def _projector_error(basis: np.ndarray, oracle: np.ndarray) -> float:
    return float(np.linalg.norm(basis @ basis.T - oracle @ oracle.T, ord="fro"))


def _scenario(seed: int, n_train: int, n_target: int, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = 10
    if name == "structured covariance shift":
        train_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
        target_variances = np.array([4.5, 4.0, 9.0, 8.0, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
        X_train = rng.normal(size=(n_train, p)) * np.sqrt(train_variances)
    elif name == "no distribution shift":
        train_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
        target_variances = train_variances
        X_train = rng.normal(size=(n_train, p)) * np.sqrt(train_variances)
    elif name == "row contamination without target shift":
        train_variances = np.array([6.0, 5.0, 2.5, 2.2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8])
        target_variances = train_variances
        X_train = rng.normal(size=(n_train, p)) * np.sqrt(train_variances)
        count = max(1, int(round(0.05 * n_train)))
        X_train[:count, 6:10] += rng.normal(loc=8.0, scale=1.5, size=(count, 4))
    else:  # pragma: no cover
        raise ValueError(name)
    X_target = rng.normal(size=(n_target, p)) * np.sqrt(target_variances)
    oracle = _basis_from_covariance(np.diag(target_variances), rank=2)
    return X_train, X_target, oracle


def run(profile: str, seed: int, repeats: int | None = None) -> list[dict[str, object]]:
    config = PROFILES[profile]
    repeats = config["repeats"] if repeats is None else repeats
    methods = [
        ("Empirical PCA", lambda X: _empirical_fit(X, 2)),
        ("RobustPCA(Cauchy)", lambda X: _robust_fit(X, 2)),
        ("DRO-PCA identity control", lambda X: _dro_fit(X, 2, "identity", 2.5)),
        ("DRO-PCA residual geometry", lambda X: _dro_fit(X, 2, "residual", 2.5)),
        ("DRO-PCA PCA-block geometry", lambda X: _dro_fit(X, 2, "pca_block", 2.5)),
    ]
    scenarios = [
        "structured covariance shift",
        "no distribution shift",
        "row contamination without target shift",
    ]
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for scenario_index, scenario in enumerate(scenarios):
            X_train, X_target, oracle = _scenario(
                seed + 1000 * repeat + 100 * scenario_index,
                config["n_train"],
                config["n_target"],
                scenario,
            )
            for method, fitter in methods:
                start = time.perf_counter()
                fitted = fitter(X_train)
                seconds = time.perf_counter() - start
                basis = np.asarray(fitted["basis"])
                location = np.asarray(fitted["location"])
                estimator = fitted["estimator"]
                row = {
                    "scenario": scenario,
                    "method": method,
                    "repeat": repeat,
                    "target_risk": _risk(X_target, location, basis),
                    "training_risk": _risk(X_train, location, basis),
                    "target_projector_error": _projector_error(basis, oracle),
                    "seconds": seconds,
                    "selected_gamma": "",
                    "exact_worst_case_risk": "",
                    "surrogate_risk_bound": "",
                }
                if isinstance(estimator, DistributionallyRobustPCA):
                    row.update(
                        selected_gamma=estimator.selected_gamma_,
                        exact_worst_case_risk=estimator.exact_worst_case_risk_,
                        surrogate_risk_bound=estimator.surrogate_risk_bound_,
                    )
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("plot generation requires robustcov[plot]") from exc
    scenario = "structured covariance shift"
    methods = sorted({str(row["method"]) for row in rows})
    means = [
        np.mean([float(row["target_risk"]) for row in rows if row["scenario"] == scenario and row["method"] == method])
        for method in methods
    ]
    order = np.argsort(means)
    methods = [methods[index] for index in order]
    means = [means[index] for index in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(methods))
    ax.barh(positions, means)
    ax.set_yticks(positions, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out target reconstruction risk (lower is better)")
    ax.set_title("Distributionally robust PCA under structured covariance shift")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    rows = run(args.profile, args.seed, args.repeats)
    if args.csv:
        write_csv(args.csv, rows)
    if args.plot:
        write_plot(args.plot, rows)
    if not args.csv:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
