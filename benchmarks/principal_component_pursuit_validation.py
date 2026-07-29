#!/usr/bin/env python3
"""Validate PCP recovery under entrywise-sparse gross corruption."""

from __future__ import annotations

import argparse

import numpy as np

from robustcov import PrincipalComponentPursuit


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / np.linalg.norm(truth, ord="fro")
    )


def _problem(seed: int, *, contaminated: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples, n_features, rank = 60, 45, 3
    left, _ = np.linalg.qr(rng.normal(size=(n_samples, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
    low_rank = left @ np.diag([20.0, 14.0, 9.0]) @ right.T
    observed = low_rank.copy()
    if contaminated:
        indices = rng.choice(
            n_samples * n_features,
            size=int(0.04 * n_samples * n_features),
            replace=False,
        )
        observed.flat[indices] += (
            rng.choice([-1.0, 1.0], size=indices.size)
            * rng.uniform(6.0, 12.0, size=indices.size)
        )
    return observed, low_rank


def _truncated_svd(matrix: np.ndarray, rank: int) -> np.ndarray:
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    return (left[:, :rank] * singular_values[:rank]) @ right[:rank]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    print(
        "scenario,repetitions,mean_svd_error,mean_pcp_error,"
        "mean_sparse_fraction,convergence_rate"
    )
    for scenario, contaminated in (
        ("clean_low_rank", False),
        ("sparse_gross_corruption", True),
    ):
        svd_errors = []
        pcp_errors = []
        sparse_fractions = []
        converged = []
        for repetition in range(args.repetitions):
            observed, truth = _problem(100 + repetition, contaminated=contaminated)
            svd_errors.append(
                _relative_frobenius(_truncated_svd(observed, 3), truth)
            )
            estimator = PrincipalComponentPursuit(tol=1e-7).fit(observed)
            pcp_errors.append(_relative_frobenius(estimator.low_rank_, truth))
            sparse_fractions.append(estimator.sparse_fraction_)
            converged.append(float(estimator.converged_))
        print(
            f"{scenario},{args.repetitions},"
            f"{np.mean(svd_errors):.8f},{np.mean(pcp_errors):.8f},"
            f"{np.mean(sparse_fractions):.8f},{np.mean(converged):.3f}"
        )


if __name__ == "__main__":
    main()
