# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust PCA subspace recovery and outlier-map diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

import robustcov as rc


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(450, 2))
    basis = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.3],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    clean = latent @ basis.T + rng.normal(scale=0.08, size=(450, 5))

    leverage = rng.normal(scale=0.2, size=(25, 5))
    leverage[:, :3] += np.array([8.0, 2.0, 3.0])

    orthogonal = rng.normal(scale=0.2, size=(25, 5))
    orthogonal[:, 3:] += rng.normal(loc=7.0, scale=1.0, size=(25, 2))

    X = np.vstack([clean, leverage, orthogonal])
    labels = np.r_[np.zeros(450, dtype=int), np.ones(50, dtype=int)]
    return X, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/robust_pca")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, labels = make_data()
    pca = rc.RobustPCA(
        n_components=2,
        estimator=rc.FastMCD(quality="balanced", random_state=0),
    ).fit(X)

    diagnostics = pca.outlier_map(X)
    print("Robust PCA")
    print("==========")
    print(f"retained components: {pca.n_components_}")
    print(
        "robust explained variance: "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )
    print(
        "median orthogonal distance, central observations: "
        f"{np.median(diagnostics[labels == 0, 1]):.3f}"
    )
    print(
        "median orthogonal distance, injected outliers: "
        f"{np.median(diagnostics[labels == 1, 1]):.3f}"
    )

    rc.plot_robust_pca_outlier_map(
        pca,
        labels=labels,
        output_path=outdir / "outlier_map.png",
        show=False,
    )


if __name__ == "__main__":
    main()
