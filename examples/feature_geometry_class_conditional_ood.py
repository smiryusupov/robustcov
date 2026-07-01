# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Class-conditional robust feature geometry for OOD-style scoring.

This example treats rows of ``Z`` as learned feature vectors.  The features could
come from a frozen image model, text encoder, autoencoder, or penultimate neural
network layer.  ``robustcov`` only sees the feature matrix.

The example compares empirical class-conditional covariance with robust
class-conditional scatter geometry when each reference class contains a small
fraction of leverage-like contaminated features.
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


def make_class_conditional_features(seed=123):
    """Create labeled feature data with class-wise leverage contamination."""
    rng = np.random.default_rng(seed)

    n_per_class = 220
    n_test_per_class = 120
    n_ood = 300
    p = 12
    n_classes = 3

    centers = np.zeros((n_classes, p))
    centers[0, 1] = -3.0
    centers[1, 1] = 0.0
    centers[2, 1] = 3.0

    leverage_direction = np.zeros(p)
    leverage_direction[0] = 1.0

    scales = np.ones(p)
    scales[0] = 0.30

    Z_train_parts = []
    y_train_parts = []

    for cls in range(n_classes):
        Z_cls = centers[cls] + rng.normal(size=(n_per_class, p)) * scales

        n_bad = int(0.10 * n_per_class)
        bad_idx = rng.choice(n_per_class, size=n_bad, replace=False)

        # Each class has leverage-like contaminated reference features along
        # the same sensitive direction.
        Z_cls[bad_idx] += 9.0 * leverage_direction + rng.normal(
            scale=0.15,
            size=(n_bad, p),
        )

        Z_train_parts.append(Z_cls)
        y_train_parts.append(np.full(n_per_class, cls, dtype=int))

    Z_train = np.vstack(Z_train_parts)
    y_train = np.concatenate(y_train_parts)

    Z_in_parts = []
    for cls in range(n_classes):
        Z_in_parts.append(
            centers[cls] + rng.normal(size=(n_test_per_class, p)) * scales
        )

    Z_in = np.vstack(Z_in_parts)

    # OOD features lie away from all class centers in the sensitive direction.
    cls_for_ood = rng.integers(0, n_classes, size=n_ood)
    Z_ood = centers[cls_for_ood] + rng.normal(size=(n_ood, p)) * scales
    Z_ood += 2.3 * leverage_direction

    Z_test = np.vstack([Z_in, Z_ood])
    y_ood = np.r_[np.zeros(Z_in.shape[0], dtype=int), np.ones(n_ood, dtype=int)]

    return Z_train, y_train, Z_test, y_ood


def summarize(name, geom, Z_test, y_ood):
    scores = geom.ood_scores(Z_test)
    return {
        "name": name,
        "scores": scores,
        "auroc": auroc(y_ood, scores),
        "mean_in": float(scores[y_ood == 0].mean()),
        "mean_ood": float(scores[y_ood == 1].mean()),
    }


def plot_scores(outdir, result, y_ood, filename):
    scores = result["scores"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(scores[y_ood == 0], bins=35, alpha=0.55, label="in-distribution")
    ax.hist(scores[y_ood == 1], bins=35, alpha=0.55, label="OOD")
    ax.set_title(result["name"])
    ax.set_xlabel("distance to nearest class")
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
    ax.set_title("Class-conditional OOD scoring from contaminated features")
    fig.tight_layout()
    fig.savefig(outdir / "class_conditional_auroc_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results/feature_geometry_class_conditional_ood",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Z_train, y_train, Z_test, y_ood = make_class_conditional_features()

    empirical = rc.ClassConditionalFeatureGeometry(
        estimator=EmpiricalFeatureCovariance(),
    ).fit(Z_train, y_train)

    robust = rc.ClassConditionalFeatureGeometry(
        estimator=rc.FastMCD(n_init=30, random_state=0),
    ).fit(Z_train, y_train)

    results = [
        summarize("Empirical class covariance", empirical, Z_test, y_ood),
        summarize("Robust FastMCD class geometry", robust, Z_test, y_ood),
    ]

    print("Class-conditional robust feature geometry")
    print("=========================================")
    print("Classes: 3")
    print("Training features per class: 220")
    print("Feature dimension: 12")
    print("Class-wise leverage contamination: 10%")
    print()
    print("Method                              AUROC    mean in-score    mean OOD-score")
    for result in results:
        print(
            f"{result['name']:33s}"
            f"{result['auroc']:7.3f}"
            f"       {result['mean_in']:10.3f}"
            f"       {result['mean_ood']:10.3f}"
        )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The score is the distance from each test feature to its nearest fitted class geometry."
    )
    print(
        "Empirical class covariance is weakened when each class reference set contains "
        "leverage-like contamination."
    )
    print(
        "Robust class-conditional geometry estimates each central class shape more stably, "
        "which improves nearest-class OOD separation in this synthetic example."
    )

    plot_scores(outdir, results[0], y_ood, "empirical_class_scores.png")
    plot_scores(outdir, results[1], y_ood, "robust_class_scores.png")
    plot_auroc_summary(outdir, results)


if __name__ == "__main__":
    main()
