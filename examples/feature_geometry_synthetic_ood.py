# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust feature geometry for contaminated learned representations.

This example treats a feature matrix as if it came from a frozen representation
model. The package does not train a neural network; it operates on feature
vectors produced elsewhere.

The example compares empirical feature covariance against robust scatter
geometry when the reference feature set contains leverage-like contamination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import robustcov as rc


class EmpiricalFeatureCovariance:
    """Small empirical covariance adapter for FeatureGeometry examples."""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = Xc.T @ Xc / X.shape[0]
        self.covariance_ = self.covariance_ + self.ridge * np.eye(X.shape[1])
        return self


def auroc(y_true, scores):
    """Compute AUROC without requiring sklearn."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    pos = y_true == 1
    neg = y_true == 0

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative examples")

    ranks = rankdata(scores)
    rank_sum_pos = ranks[pos].sum()

    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_feature_data(seed=42):
    """Create a contaminated reference feature space and clean/OOD test features."""
    rng = np.random.default_rng(seed)

    n_ref = 600
    n_in = 300
    n_ood = 300
    p = 16

    direction = np.zeros(p)
    direction[0] = 1.0

    # Central features have low variance in the first direction.
    # This makes that direction informative for OOD unless empirical covariance
    # is inflated by leverage contamination.
    scales = np.ones(p)
    scales[0] = 0.35

    Z_ref = rng.normal(size=(n_ref, p)) * scales

    n_bad = int(0.12 * n_ref)
    bad_idx = rng.choice(n_ref, size=n_bad, replace=False)

    # Leverage-like contamination in the same direction as the later OOD shift.
    Z_ref[bad_idx] += 10.0 * direction + rng.normal(scale=0.15, size=(n_bad, p))

    Z_in = rng.normal(size=(n_in, p)) * scales

    # OOD samples move in the sensitive direction.
    Z_ood = rng.normal(size=(n_ood, p)) * scales + 2.5 * direction

    Z_test = np.vstack([Z_in, Z_ood])
    y_test = np.r_[np.zeros(n_in, dtype=int), np.ones(n_ood, dtype=int)]

    return Z_ref, Z_test, y_test


def summarize_method(name, geom, Z_test, y_test):
    scores = geom.mahalanobis_scores(Z_test)
    return {
        "name": name,
        "scores": scores,
        "auroc": auroc(y_test, scores),
        "mean_in": float(scores[y_test == 0].mean()),
        "mean_ood": float(scores[y_test == 1].mean()),
    }


def plot_scores(outdir, result, y_test, filename):
    scores = result["scores"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(scores[y_test == 0], bins=35, alpha=0.55, label="in-distribution")
    ax.hist(scores[y_test == 1], bins=35, alpha=0.55, label="OOD")
    ax.set_title(result["name"])
    ax.set_xlabel("Mahalanobis score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=180)
    plt.close(fig)


def plot_auroc_summary(outdir, results):
    labels = [r["name"].replace(" geometry", "") for r in results]
    values = [r["auroc"] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("OOD detection from contaminated reference features")
    fig.tight_layout()
    fig.savefig(outdir / "auroc_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/feature_geometry_synthetic_ood")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Z_ref, Z_test, y_test = make_feature_data()

    empirical = rc.FeatureGeometry(
        estimator=EmpiricalFeatureCovariance(),
    ).fit(Z_ref)

    fastmcd = rc.FeatureGeometry(
        estimator=rc.FastMCD(n_init=40, random_state=0),
    ).fit(Z_ref)
    results = [
        summarize_method("Empirical covariance", empirical, Z_test, y_test),
        summarize_method("Robust FastMCD geometry", fastmcd, Z_test, y_test),
    ]

    print("Robust feature geometry: synthetic OOD example")
    print("================================================")
    print("Reference features: 600")
    print("Feature dimension:   16")
    print("Reference leverage contamination: 12%")
    print()
    print("Method                       AUROC    mean in-score    mean OOD-score")
    for result in results:
        print(
            f"{result['name']:28s}"
            f"{result['auroc']:7.3f}"
            f"       {result['mean_in']:10.3f}"
            f"       {result['mean_ood']:10.3f}"
        )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The empirical covariance is fitted on a reference feature set that contains "
        "leverage-like contamination."
    )
    print(
        "This inflates variance along the contaminated direction and weakens "
        "Mahalanobis-based OOD scores."
    )
    print(
        "Robust feature geometry estimates the central feature-space shape more "
        "stably and restores separation in this synthetic example."
    )

    plot_scores(outdir, results[0], y_test, "empirical_scores.png")
    plot_scores(outdir, results[1], y_test, "fastmcd_scores.png")
    plot_auroc_summary(outdir, results)


if __name__ == "__main__":
    main()
