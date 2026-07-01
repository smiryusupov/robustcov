# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Robust feature geometry for similarity kernels.

This synthetic example treats feature vectors as embeddings produced by an
upstream model. robustcov only sees the feature matrix.

A fitted feature geometry induces an RBF similarity kernel through robust
Mahalanobis distances. The diagnostic asks whether this kernel remains sensitive
to a shift in a low-variance feature direction when the reference set itself is
contaminated by leverage-like points.

The intended pattern is:

    clean empirical RBF geometry separates reference and shifted features;
    contaminated empirical RBF geometry can become too similar to shifted data;
    robust contaminated RBF geometry remains close to clean-reference behavior.
"""

from __future__ import annotations

import numpy as np

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


def mean_topk_similarity(K, k):
    """Average top-k kernel similarity per query row."""
    K = np.asarray(K, dtype=float)
    if K.shape[1] < k:
        raise ValueError("Need at least k reference points")

    topk = np.partition(K, kth=K.shape[1] - k, axis=1)[:, -k:]
    return float(topk.mean())


def summarize_kernel(name, geometry, X_ref_eval, X_new, X_ref_anchor, length_scale, top_k):
    """Summarize reference/reference and new/reference kernel similarities."""
    K_same = geometry.rbf_kernel(
        X_ref_eval,
        X_ref_anchor,
        length_scale=length_scale,
    )
    K_new = geometry.rbf_kernel(
        X_new,
        X_ref_anchor,
        length_scale=length_scale,
    )

    same_topk = mean_topk_similarity(K_same, top_k)
    new_topk = mean_topk_similarity(K_new, top_k)

    return {
        "name": name,
        "same_topk": same_topk,
        "new_topk": new_topk,
        "contrast": same_topk - new_topk,
    }


def main():
    rng = np.random.default_rng(123)

    n_features = 20
    n_fit = 800
    n_eval = 400
    n_anchor = 400

    contamination_fraction = 0.12
    leverage_strength = 18.0
    shift_strength = 4.0

    top_k = 10
    length_scale = np.sqrt(2.0 * n_features)

    X_ref_fit, scales = make_reference_features(rng, n_fit, n_features)
    X_ref_eval, _ = make_reference_features(rng, n_eval, n_features)

    drift_direction = np.zeros(n_features)
    drift_direction[-1] = 1.0
    drift_scale = scales[-1]

    X_new, _ = make_reference_features(rng, n_eval, n_features)
    X_new = X_new + shift_strength * drift_scale * drift_direction

    X_ref_bad, n_bad = contaminate_reference(
        X_ref_fit,
        drift_direction,
        drift_scale,
        fraction=contamination_fraction,
        strength=leverage_strength,
        seed=321,
    )

    X_ref_anchor = X_ref_fit[:n_anchor]

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
        summarize_kernel(
            name,
            geom,
            X_ref_eval,
            X_new,
            X_ref_anchor,
            length_scale,
            top_k,
        )
        for name, geom in methods
    ]

    by_name = {result["name"]: result for result in results}

    empirical_contrast_drop = (
        by_name["Empirical clean references"]["contrast"]
        - by_name["Empirical contaminated"]["contrast"]
    )
    robust_contrast_drop = (
        by_name["Robust FastMCD clean refs"]["contrast"]
        - by_name["Robust FastMCD contaminated"]["contrast"]
    )
    robust_contrast_gain = (
        by_name["Robust FastMCD contaminated"]["contrast"]
        - by_name["Empirical contaminated"]["contrast"]
    )

    print("Robust feature geometry: similarity-kernel diagnostic")
    print("=====================================================")
    print(f"reference fit samples:        {n_fit}")
    print(f"reference anchor samples:     {n_anchor}")
    print(f"held-out reference samples:   {n_eval}")
    print(f"shifted/new samples:          {n_eval}")
    print(f"feature dimension:            {n_features}")
    print(f"reference contamination:      {contamination_fraction:.0%}")
    print(f"contaminated reference count: {n_bad}")
    print("shift direction:              low-variance feature axis")
    print(f"leverage strength:            {leverage_strength}")
    print(f"new-distribution shift:       {shift_strength}")
    print(f"RBF length scale:             {length_scale:.3f}")
    print(f"top-k similarity summary:     {top_k}")
    print()
    print(
        "Method                              "
        "same-ref top-k   new-ref top-k   contrast"
    )

    for result in results:
        print(
            f"{result['name']:33s}"
            f"{result['same_topk']:14.3f}"
            f"{result['new_topk']:15.3f}"
            f"{result['contrast']:11.3f}"
        )

    print()
    print("Diagnostic summary")
    print("------------------")
    print(
        "empirical contamination contrast drop: "
        f"{empirical_contrast_drop:+.3f}"
    )
    print(
        "robust contamination contrast drop:    "
        f"{robust_contrast_drop:+.3f}"
    )
    print(
        "robust contrast gain over contaminated empirical: "
        f"{robust_contrast_gain:+.3f}"
    )

    if (
        empirical_contrast_drop > 0.08
        and robust_contrast_drop < 0.05
        and robust_contrast_gain > 0.08
    ):
        print(
            "pattern: empirical RBF geometry loses contrast under leverage "
            "contamination, while robust feature geometry preserves kernel "
            "sensitivity to the shifted distribution."
        )
    else:
        print(
            "pattern: mixed result; inspect length scale, contamination strength, "
            "and shift direction before making a claim."
        )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The RBF kernel is computed from distances induced by a fitted feature "
        "geometry."
    )
    print(
        "When empirical covariance is contaminated in a sensitive low-variance "
        "direction, shifted features can remain spuriously similar to reference "
        "features."
    )
    print(
        "A robust scatter estimate gives a more stable feature metric, so the "
        "kernel similarity to shifted features drops as expected."
    )


if __name__ == "__main__":
    main()
