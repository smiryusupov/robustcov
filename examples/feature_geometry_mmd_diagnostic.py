# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""MMD with a robust feature-space metric.

This synthetic example treats feature vectors as embeddings produced by an
upstream model. robustcov only sees the feature matrix.

A fitted FeatureGeometry induces an RBF kernel through Mahalanobis-style
distances.  We then compute MMD with that kernel between an old/reference
feature distribution and a new/current feature distribution.

This is not a new MMD theory contribution. It is ordinary kernel MMD using a
kernel whose metric is estimated robustly from reference features.

The intended pattern is:

    clean empirical metric gives a visible drift MMD;
    contaminated empirical metric can be blinded by leverage contamination;
    robust contaminated metric remains close to clean-reference behavior.
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


def mmd2_biased(geometry, X, Y, *, length_scale):
    """Biased nonnegative MMD^2 estimate with the geometry-induced RBF kernel."""
    Kxx = geometry.rbf_kernel(X, X, length_scale=length_scale)
    Kyy = geometry.rbf_kernel(Y, Y, length_scale=length_scale)
    Kxy = geometry.rbf_kernel(X, Y, length_scale=length_scale)

    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def summarize_mmd(name, geometry, X_ref_a, X_ref_b, X_new, length_scale):
    """Compare reference/reference and reference/new MMD under one geometry."""
    null_mmd2 = mmd2_biased(
        geometry,
        X_ref_a,
        X_ref_b,
        length_scale=length_scale,
    )
    drift_mmd2 = mmd2_biased(
        geometry,
        X_ref_a,
        X_new,
        length_scale=length_scale,
    )

    return {
        "name": name,
        "null_mmd2": null_mmd2,
        "drift_mmd2": drift_mmd2,
        "excess_mmd2": drift_mmd2 - null_mmd2,
    }


def main():
    rng = np.random.default_rng(123)

    n_features = 20
    n_fit = 800
    n_eval = 300

    contamination_fraction = 0.12
    leverage_strength = 18.0
    shift_strength = 4.0

    length_scale = np.sqrt(2.0 * n_features)

    X_ref_fit, scales = make_reference_features(rng, n_fit, n_features)
    X_ref_a, _ = make_reference_features(rng, n_eval, n_features)
    X_ref_b, _ = make_reference_features(rng, n_eval, n_features)

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
        summarize_mmd(
            name,
            geom,
            X_ref_a,
            X_ref_b,
            X_new,
            length_scale,
        )
        for name, geom in methods
    ]

    by_name = {result["name"]: result for result in results}

    empirical_excess_drop = (
        by_name["Empirical clean references"]["excess_mmd2"]
        - by_name["Empirical contaminated"]["excess_mmd2"]
    )
    robust_excess_drop = (
        by_name["Robust FastMCD clean refs"]["excess_mmd2"]
        - by_name["Robust FastMCD contaminated"]["excess_mmd2"]
    )
    robust_gain = (
        by_name["Robust FastMCD contaminated"]["excess_mmd2"]
        - by_name["Empirical contaminated"]["excess_mmd2"]
    )

    print("Robust feature geometry: MMD diagnostic")
    print("=======================================")
    print(f"reference fit samples:        {n_fit}")
    print(f"reference MMD samples:        {n_eval} + {n_eval}")
    print(f"new/current samples:          {n_eval}")
    print(f"feature dimension:            {n_features}")
    print(f"reference contamination:      {contamination_fraction:.0%}")
    print(f"contaminated reference count: {n_bad}")
    print("shift direction:              low-variance feature axis")
    print(f"leverage strength:            {leverage_strength}")
    print(f"new-distribution shift:       {shift_strength}")
    print(f"RBF length scale:             {length_scale:.3f}")
    print()
    print(
        "Method                              "
        "ref-ref MMD^2   ref-new MMD^2   excess MMD^2"
    )

    for result in results:
        print(
            f"{result['name']:33s}"
            f"{result['null_mmd2']:13.4f}"
            f"{result['drift_mmd2']:16.4f}"
            f"{result['excess_mmd2']:15.4f}"
        )

    print()
    print("Diagnostic summary")
    print("------------------")
    print(
        "empirical contamination excess-MMD drop: "
        f"{empirical_excess_drop:+.4f}"
    )
    print(
        "robust contamination excess-MMD drop:    "
        f"{robust_excess_drop:+.4f}"
    )
    print(
        "robust excess-MMD gain over contaminated empirical: "
        f"{robust_gain:+.4f}"
    )

    if (
        empirical_excess_drop > 0.015
        and robust_excess_drop < 0.005
        and robust_gain > 0.015
    ):
        print(
            "pattern: empirical metric MMD is weakened by leverage contamination, "
            "while MMD with a robust feature-space metric preserves sensitivity "
            "to the shifted distribution."
        )
    else:
        print(
            "pattern: mixed result; inspect kernel length scale, contamination "
            "strength, and shift direction before making a claim."
        )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "This is ordinary kernel MMD computed with an RBF kernel induced by a "
        "fitted feature geometry."
    )
    print(
        "When empirical covariance is contaminated in a sensitive low-variance "
        "direction, the induced kernel can make reference and shifted "
        "distributions look too similar."
    )
    print(
        "A robust scatter estimate gives a more stable feature-space metric, so "
        "the MMD remains sensitive to the shift."
    )


if __name__ == "__main__":
    main()
