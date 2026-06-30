# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust feature geometry for representation drift detection.

This synthetic example treats feature vectors as embeddings produced by an
upstream model. robustcov only sees the feature matrix.

The diagnostic asks whether a fitted reference geometry can detect a shift in a
low-variance feature direction when the reference set itself contains
leverage-like contamination.

The intended pattern is:

    clean empirical geometry detects the drift;
    contaminated empirical geometry can be blinded by leverage contamination;
    robust contaminated geometry remains close to clean-reference behavior.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import robustcov as rc


class EmpiricalFeatureCovariance:
    """Small empirical covariance adapter for FeatureGeometry examples."""

    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = Xc.T @ Xc / X.shape[0]
        self.covariance_ = self.covariance_ + self.ridge * np.eye(X.shape[1])
        return self


def auroc(y_true, scores):
    """Compute AUROC without depending on sklearn."""
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


def make_reference_features(rng, n_samples, n_features):
    """Make anisotropic representation features."""
    scales = np.geomspace(1.5, 0.15, n_features)
    X = rng.normal(size=(n_samples, n_features)) * scales
    return X, scales


def contaminate_reference(X, direction, scale, fraction, strength, seed):
    """Add one-sided leverage contamination to the reference features."""
    rng = np.random.default_rng(seed)

    X_bad = X.copy()
    n_bad = int(round(fraction * X.shape[0]))
    idx = rng.choice(X.shape[0], size=n_bad, replace=False)

    X_bad[idx] = X_bad[idx] + strength * scale * direction

    return X_bad, n_bad


def summarize_geometry(name, geometry, X_ref_eval, X_new):
    """Score held-out reference and new features under one geometry."""
    ref_scores = geometry.decision_function(X_ref_eval)
    new_scores = geometry.decision_function(X_new)

    y = np.r_[
        np.zeros(ref_scores.shape[0], dtype=int),
        np.ones(new_scores.shape[0], dtype=int),
    ]
    scores = np.r_[ref_scores, new_scores]

    return {
        "name": name,
        "auroc": auroc(y, scores),
        "mean_ref": float(ref_scores.mean()),
        "mean_new": float(new_scores.mean()),
        "q90_ref": float(np.quantile(ref_scores, 0.90)),
        "q90_new": float(np.quantile(new_scores, 0.90)),
    }


def main():
    rng = np.random.default_rng(123)

    n_features = 20
    n_fit = 800
    n_eval = 500

    contamination_fraction = 0.12
    leverage_strength = 18.0
    drift_strength = 4.0

    X_ref_fit, scales = make_reference_features(rng, n_fit, n_features)
    X_ref_eval, _ = make_reference_features(rng, n_eval, n_features)

    drift_direction = np.zeros(n_features)
    drift_direction[-1] = 1.0
    drift_scale = scales[-1]

    X_new, _ = make_reference_features(rng, n_eval, n_features)
    X_new = X_new + drift_strength * drift_scale * drift_direction

    X_ref_bad, n_bad = contaminate_reference(
        X_ref_fit,
        drift_direction,
        drift_scale,
        fraction=contamination_fraction,
        strength=leverage_strength,
        seed=321,
    )

    methods = [
        (
            "Empirical clean references",
            rc.FeatureGeometry(
                estimator=EmpiricalFeatureCovariance(),
            ).fit(X_ref_fit),
        ),
        (
            "Empirical contaminated",
            rc.FeatureGeometry(
                estimator=EmpiricalFeatureCovariance(),
            ).fit(X_ref_bad),
        ),
        (
            "Robust FastMCD clean refs",
            rc.FeatureGeometry(
                estimator=rc.FastMCD(n_init=20, random_state=0),
            ).fit(X_ref_fit),
        ),
        (
            "Robust FastMCD contaminated",
            rc.FeatureGeometry(
                estimator=rc.FastMCD(n_init=20, random_state=0),
            ).fit(X_ref_bad),
        ),
    ]

    results = [
        summarize_geometry(name, geom, X_ref_eval, X_new)
        for name, geom in methods
    ]

    by_name = {result["name"]: result for result in results}

    empirical_drop = (
        by_name["Empirical clean references"]["auroc"]
        - by_name["Empirical contaminated"]["auroc"]
    )
    robust_drop = (
        by_name["Robust FastMCD clean refs"]["auroc"]
        - by_name["Robust FastMCD contaminated"]["auroc"]
    )
    robust_gain = (
        by_name["Robust FastMCD contaminated"]["auroc"]
        - by_name["Empirical contaminated"]["auroc"]
    )

    print("Robust feature geometry: drift detection diagnostic")
    print("===================================================")
    print(f"reference fit samples:        {n_fit}")
    print(f"held-out reference samples:   {n_eval}")
    print(f"new/current samples:          {n_eval}")
    print(f"feature dimension:            {n_features}")
    print(f"reference contamination:      {contamination_fraction:.0%}")
    print(f"contaminated reference count: {n_bad}")
    print("drift direction:              low-variance feature axis")
    print(f"leverage strength:            {leverage_strength}")
    print(f"new-distribution shift:       {drift_strength}")
    print()
    print(
        "Method                              "
        "AUROC    mean ref    mean new     q90 ref     q90 new"
    )

    for result in results:
        print(
            f"{result['name']:33s}"
            f"{result['auroc']:7.3f}"
            f"  {result['mean_ref']:10.3f}"
            f"  {result['mean_new']:10.3f}"
            f"  {result['q90_ref']:10.3f}"
            f"  {result['q90_new']:10.3f}"
        )

    print()
    print("Diagnostic summary")
    print("------------------")
    print(f"empirical contamination drop:       {empirical_drop:+.3f} AUROC")
    print(f"robust contamination drop:          {robust_drop:+.3f} AUROC")
    print(f"robust gain over contaminated empirical: {robust_gain:+.3f} AUROC")

    if empirical_drop > 0.20 and robust_drop < 0.05 and robust_gain > 0.20:
        print(
            "pattern: empirical reference geometry is blinded by leverage "
            "contamination, while robust geometry preserves drift sensitivity."
        )
    else:
        print(
            "pattern: mixed result; inspect contamination strength and drift "
            "direction before making a claim."
        )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The example treats feature vectors as learned representations from an "
        "upstream model."
    )
    print(
        "The reference distribution is contaminated in the same low-variance "
        "direction where the new distribution later shifts."
    )
    print(
        "Empirical covariance inflates that direction and can make the drift look "
        "ordinary. Robust feature geometry estimates the central reference shape "
        "more stably."
    )


if __name__ == "__main__":
    main()
