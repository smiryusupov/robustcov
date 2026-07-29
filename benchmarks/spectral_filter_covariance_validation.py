#!/usr/bin/env python3
"""Deterministic validation for spectral filtering under adversarial row attacks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robustcov.experimental import SpectralFilteringCovariance


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / np.linalg.norm(truth, ord="fro")
    )


def _replicate(seed: int, *, contaminated: bool) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, p = 900, 10
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    truth = Q @ np.diag(np.geomspace(4.0, 0.6, p)) @ Q.T
    X = rng.multivariate_normal(np.zeros(p), truth, size=n)
    outlier_mask = np.zeros(n, dtype=bool)
    if contaminated:
        indices = rng.choice(n, size=90, replace=False)
        outlier_mask[indices] = True
        direction = rng.normal(size=p)
        direction /= np.linalg.norm(direction)
        signs = rng.choice([-1.0, 1.0], size=indices.size)
        X[indices] = (
            signs[:, None] * 11.0 * direction
            + rng.normal(scale=0.35, size=(indices.size, p))
        )

    empirical = np.cov(X, rowvar=False, bias=True)
    filtered = SpectralFilteringCovariance(
        contamination=0.1,
        power_iterations=15,
        random_state=0,
    ).fit(X)
    return {
        "empirical_error": _relative_frobenius(empirical, truth),
        "filtered_error": _relative_frobenius(filtered.covariance_, truth),
        "removed_fraction": float(filtered.n_removed_ / n),
        "outlier_recall": (
            float(np.mean(~filtered.support_[outlier_mask]))
            if contaminated
            else 0.0
        ),
    }


def run_validation(repetitions: int = 5) -> list[dict[str, float | int | str]]:
    rows = []
    for contaminated, scenario in (
        (False, "clean_gaussian"),
        (True, "rank_one_adversarial_rows"),
    ):
        values = [_replicate(seed, contaminated=contaminated) for seed in range(repetitions)]
        rows.append(
            {
                "scenario": scenario,
                "repetitions": repetitions,
                "mean_empirical_error": float(np.mean([v["empirical_error"] for v in values])),
                "mean_filtered_error": float(np.mean([v["filtered_error"] for v in values])),
                "mean_removed_fraction": float(np.mean([v["removed_fraction"] for v in values])),
                "mean_outlier_recall": float(np.mean([v["outlier_recall"] for v in values])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = run_validation(args.repetitions)
    print(
        "scenario,repetitions,mean_empirical_error,mean_filtered_error,"
        "mean_removed_fraction,mean_outlier_recall"
    )
    for row in rows:
        print(
            f"{row['scenario']},{row['repetitions']},"
            f"{row['mean_empirical_error']:.6f},{row['mean_filtered_error']:.6f},"
            f"{row['mean_removed_fraction']:.6f},{row['mean_outlier_recall']:.6f}"
        )
    attack = rows[1]
    if not attack["mean_filtered_error"] < 0.35 * attack["mean_empirical_error"]:
        raise SystemExit("spectral filtering did not improve enough over empirical covariance")
    if not attack["mean_outlier_recall"] >= 0.8:
        raise SystemExit("spectral filtering did not identify enough attack rows")
    clean = rows[0]
    if not clean["mean_removed_fraction"] <= 0.01:
        raise SystemExit("clean Gaussian validation removed too many rows")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
