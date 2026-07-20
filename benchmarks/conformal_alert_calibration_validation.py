#!/usr/bin/env python3
"""Validate split-conformal alert calibration and contaminated-reference behavior."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robustcov as rc


def run_validation(
    *,
    seed: int = 0,
    repetitions: int = 200,
    n_calibration: int = 499,
    n_test: int = 2_000,
    alpha: float = 0.05,
    contamination_fraction: float = 0.05,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    clean_rates = []
    contaminated_rates = []
    for _ in range(repetitions):
        calibration = rng.normal(size=n_calibration)
        test_scores = rng.normal(size=n_test)
        clean = rc.ConformalAlertCalibrator(alpha=alpha).fit(calibration)
        clean_rates.append(float(np.mean(clean.predict_alerts(test_scores))))

        n_contaminated = int(np.floor(contamination_fraction * n_calibration))
        contaminated_scores = calibration.copy()
        if n_contaminated:
            selected = rng.choice(n_calibration, size=n_contaminated, replace=False)
            contaminated_scores[selected] += 8.0
        contaminated = rc.ConformalAlertCalibrator(alpha=alpha).fit(
            contaminated_scores
        )
        contaminated_rates.append(
            float(np.mean(contaminated.predict_alerts(test_scores)))
        )

    clean_array = np.asarray(clean_rates)
    contaminated_array = np.asarray(contaminated_rates)
    return [
        {
            "scenario": "clean_exchangeable_reference",
            "mean_alert_rate": float(np.mean(clean_array)),
            "q95_alert_rate": float(np.quantile(clean_array, 0.95)),
            "target_alpha": float(alpha),
            "repetitions": int(repetitions),
            "n_calibration": int(n_calibration),
            "n_test": int(n_test),
        },
        {
            "scenario": "upper_tail_contaminated_reference",
            "mean_alert_rate": float(np.mean(contaminated_array)),
            "q95_alert_rate": float(np.quantile(contaminated_array, 0.95)),
            "target_alpha": float(alpha),
            "repetitions": int(repetitions),
            "n_calibration": int(n_calibration),
            "n_test": int(n_test),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--n-calibration", type=int, default=499)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--contamination-fraction", type=float, default=0.05)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = run_validation(
        seed=args.seed,
        repetitions=args.repetitions,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        alpha=args.alpha,
        contamination_fraction=args.contamination_fraction,
    )
    print("scenario,mean_alert_rate,q95_alert_rate,target_alpha")
    for row in rows:
        print(
            f"{row['scenario']},{row['mean_alert_rate']:.6f},"
            f"{row['q95_alert_rate']:.6f},{row['target_alpha']:.6f}"
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
