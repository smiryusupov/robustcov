# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Practical embedding-monitoring workflow.

This example shows how robust feature geometry can be used as a drop-in
monitoring layer for feature or embedding matrices.

The setting is intentionally practical:

    X_ref: reference embeddings from a baseline time window
    X_new: embeddings from a new time window
    X_ref may be contaminated by old shifted batches or noncentral examples

The workflow compares MMD-style drift signals under empirical covariance
geometry and robust FastMCD geometry. A threshold is calibrated from reference
splits, mimicking a simple production monitoring rule.
"""

from __future__ import annotations

import numpy as np

import robustcov as rc


class EmpiricalFeatureCovariance:
    """Small empirical covariance adapter."""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = Xc.T @ Xc / X.shape[0]
        self.covariance_ = self.covariance_ + self.ridge * np.eye(X.shape[1])
        return self


def mmd2_biased_from_kernel(Kxx, Kyy, Kxy):
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def metric_mmd2(geometry, X, Y, *, length_scale):
    Kxx = geometry.rbf_kernel(X, X, length_scale=length_scale)
    Kyy = geometry.rbf_kernel(Y, Y, length_scale=length_scale)
    Kxy = geometry.rbf_kernel(X, Y, length_scale=length_scale)
    return mmd2_biased_from_kernel(Kxx, Kyy, Kxy)


def reference_split_threshold(
    geometry,
    X_ref,
    *,
    length_scale,
    block_size,
    n_splits,
    quantile,
    random_state,
):
    """Calibrate a simple reference-only MMD threshold."""
    rng = np.random.default_rng(random_state)

    scores = []
    n = X_ref.shape[0]

    for _ in range(n_splits):
        idx = rng.choice(n, size=2 * block_size, replace=False)
        A = X_ref[idx[:block_size]]
        B = X_ref[idx[block_size:]]
        scores.append(metric_mmd2(geometry, A, B, length_scale=length_scale))

    scores = np.asarray(scores, dtype=float)

    return {
        "threshold": float(np.quantile(scores, quantile)),
        "median": float(np.median(scores)),
        "scores": scores,
    }


def make_embedding_windows(
    *,
    n_ref=1000,
    n_new=400,
    n_features=20,
    contamination_fraction=0.16,
    contamination_strength=2.7,
    drift_strength=0.70,
    random_state=123,
):
    """Create a monitoring problem with a contaminated reference window.

    The first coordinate is a low-variance feature direction. The new batch
    drifts in that direction. The contaminated reference window contains
    symmetric old shifted batches in the same direction, which inflates empirical
    covariance but should be downweighted by robust scatter.
    """
    rng = np.random.default_rng(random_state)

    variances = np.linspace(0.08, 1.0, n_features)
    variances[0] = 0.015
    scale = np.sqrt(variances)

    X_ref_clean = rng.normal(size=(n_ref, n_features)) * scale
    X_new_clean = rng.normal(size=(n_new, n_features)) * scale

    X_new_drift = rng.normal(size=(n_new, n_features)) * scale
    X_new_drift[:, 0] += drift_strength

    X_ref_bad = X_ref_clean.copy()

    n_bad = int(round(contamination_fraction * n_ref))
    bad_idx = rng.choice(n_ref, size=n_bad, replace=False)

    half = n_bad // 2
    X_ref_bad[bad_idx[:half], 0] += contamination_strength
    X_ref_bad[bad_idx[half:], 0] -= contamination_strength

    bad_mask = np.zeros(n_ref, dtype=bool)
    bad_mask[bad_idx] = True

    return {
        "X_ref_clean": X_ref_clean,
        "X_ref_bad": X_ref_bad,
        "X_new_clean": X_new_clean,
        "X_new_drift": X_new_drift,
        "n_bad": n_bad,
        "bad_mask": bad_mask,
        "n_features": n_features,
        "contamination_fraction": contamination_fraction,
        "contamination_strength": contamination_strength,
        "drift_strength": drift_strength,
    }



def central_reference_anchor(geometry, X_ref, *, keep_fraction, bad_mask=None):
    """Keep the central reference embeddings under the fitted geometry."""
    scores = geometry.decision_function(X_ref)
    cutoff = float(np.quantile(scores, keep_fraction))
    keep = scores <= cutoff

    if bad_mask is None:
        bad_kept = 0
        bad_total = 0
    else:
        bad_mask = np.asarray(bad_mask, dtype=bool)
        bad_kept = int((keep & bad_mask).sum())
        bad_total = int(bad_mask.sum())

    return {
        "X_anchor": X_ref[keep],
        "kept": int(keep.sum()),
        "cutoff": cutoff,
        "bad_kept": bad_kept,
        "bad_total": bad_total,
    }


def monitor_row(
    name,
    geometry,
    X_ref,
    X_new_clean,
    X_new_drift,
    *,
    length_scale,
    keep_fraction,
    bad_mask=None,
):
    anchor = central_reference_anchor(
        geometry,
        X_ref,
        keep_fraction=keep_fraction,
        bad_mask=bad_mask,
    )
    X_anchor = anchor["X_anchor"]

    calibration = reference_split_threshold(
        geometry,
        X_anchor,
        length_scale=length_scale,
        block_size=200,
        n_splits=80,
        quantile=0.95,
        random_state=321,
    )

    clean_score = metric_mmd2(
        geometry,
        X_anchor[:400],
        X_new_clean,
        length_scale=length_scale,
    )
    drift_score = metric_mmd2(
        geometry,
        X_anchor[:400],
        X_new_drift,
        length_scale=length_scale,
    )

    threshold = calibration["threshold"]

    return {
        "name": name,
        "threshold": threshold,
        "baseline": calibration["median"],
        "clean_score": clean_score,
        "drift_score": drift_score,
        "clean_alert": clean_score > threshold,
        "drift_alert": drift_score > threshold,
        "drift_over_threshold": drift_score / threshold if threshold > 0 else np.inf,
        "kept": anchor["kept"],
        "bad_kept": anchor["bad_kept"],
        "bad_total": anchor["bad_total"],
    }


def main():
    data = make_embedding_windows()

    X_ref_clean = data["X_ref_clean"]
    X_ref_bad = data["X_ref_bad"]
    X_new_clean = data["X_new_clean"]
    X_new_drift = data["X_new_drift"]
    bad_mask = data["bad_mask"]

    length_scale = float(np.sqrt(data["n_features"]))
    keep_fraction = 0.80

    methods = [
        (
            "Empirical clean reference",
            X_ref_clean,
            None,
            rc.FeatureGeometry(
                estimator=EmpiricalFeatureCovariance(),
            ).fit(X_ref_clean),
        ),
        (
            "Empirical contaminated",
            X_ref_bad,
            bad_mask,
            rc.FeatureGeometry(
                estimator=EmpiricalFeatureCovariance(),
            ).fit(X_ref_bad),
        ),
        (
            "Robust FastMCD clean ref",
            X_ref_clean,
            None,
            rc.FeatureGeometry(
                estimator=rc.FastMCD(n_init=20, random_state=0),
            ).fit(X_ref_clean),
        ),
        (
            "Robust FastMCD contaminated",
            X_ref_bad,
            bad_mask,
            rc.FeatureGeometry(
                estimator=rc.FastMCD(n_init=20, random_state=0),
            ).fit(X_ref_bad),
        ),
    ]

    rows = [
        monitor_row(
            name,
            geom,
            X_cal,
            X_new_clean,
            X_new_drift,
            length_scale=length_scale,
            keep_fraction=keep_fraction,
            bad_mask=mask,
        )
        for name, X_cal, mask, geom in methods
    ]

    by_name = {row["name"]: row for row in rows}
    robust_gain = (
        by_name["Robust FastMCD contaminated"]["drift_score"]
        - by_name["Empirical contaminated"]["drift_score"]
    )

    print("Practical embedding-monitoring workflow")
    print("=======================================")
    print(f"reference embeddings:          {X_ref_clean.shape[0]}")
    print(f"new-batch embeddings:          {X_new_clean.shape[0]}")
    print(f"feature dimension:             {data['n_features']}")
    print(f"reference contamination:       {data['contamination_fraction']:.0%}")
    print(f"contaminated reference count:  {data['n_bad']}")
    print(f"contamination strength:        {data['contamination_strength']:.2f}")
    print(f"new-batch drift strength:      {data['drift_strength']:.2f}")
    print(f"MMD length scale:              {length_scale:.3f}")
    print(f"central reference kept:        {keep_fraction:.0%}")

    print()
    print(
        "Method                           "
        "kept  bad kept  baseline   thresh95   clean MMD   drift MMD   clean alert   drift alert   drift/thresh"
    )

    for row in rows:
        print(
            f"{row['name']:31s}"
            f"{row['kept']:5d}"
            f"{row['bad_kept']:10d}"
            f"{row['baseline']:10.4f}"
            f"{row['threshold']:11.4f}"
            f"{row['clean_score']:11.4f}"
            f"{row['drift_score']:12.4f}"
            f"{str(row['clean_alert']):>14s}"
            f"{str(row['drift_alert']):>14s}"
            f"{row['drift_over_threshold']:14.2f}"
        )

    print()
    print("Monitoring interpretation")
    print("-------------------------")
    print(
        "The workflow first keeps a central reference anchor under the fitted "
        "geometry. The clean new batch should stay near the reference-split "
        "threshold; the drifted batch should exceed it."
    )
    print(
        "When the reference window is contaminated, empirical covariance geometry "
        "can absorb the shifted direction and reduce the MMD drift signal."
    )
    print(
        "Robust feature geometry is intended to keep the monitoring metric closer "
        "to the central reference geometry."
    )
    print()
    print(
        "Robust contaminated drift-MMD gain over empirical contaminated: "
        f"{robust_gain:+.4f}"
    )


if __name__ == "__main__":
    main()
