"""Example: ML use cases for SPD geometry utilities.

This script shows practical ways to use ``robustcov.geometry``:

1. Detect covariance/regime drift across feature windows.
2. Compare empirical and robust scatter under contamination.
3. Quantify estimator stability under contaminated inputs.
4. Turn robust scatter into a robust similarity score.
5. Inspect a regularized Tyler path on the SPD cone.

The data are synthetic, but the patterns mirror common ML workflows: monitoring
feature drift, comparing covariance estimates, stress-testing preprocessing
geometry, and using robust input similarity.
"""

from __future__ import annotations

import numpy as np

import robustcov as rc
import robustcov.geometry as rcg


def make_window(rng, n_samples=160, regime="baseline", contamination=0.0):
    if regime == "baseline":
        A = np.array([[1.0, 0.55], [0.0, 0.45]])
    elif regime == "shifted":
        A = np.array([[1.0, -0.15], [0.0, 1.10]])
    else:
        raise ValueError("unknown regime")

    X = rng.normal(size=(n_samples, 2)) @ A.T

    n_bad = int(contamination * n_samples)
    if n_bad:
        bad = np.column_stack(
            [
                rng.normal(0.0, 0.5, size=n_bad),
                rng.normal(0.0, 5.0, size=n_bad),
            ]
        )
        X[:n_bad] = bad

    return X


def contaminate_same_sample(rng, X, contamination=0.10):
    X_bad = X.copy()
    n_bad = int(contamination * X.shape[0])
    X_bad[:n_bad] = np.column_stack(
        [
            rng.normal(0.0, 0.5, size=n_bad),
            rng.normal(0.0, 5.0, size=n_bad),
        ]
    )
    return X_bad


def robust_scatter(X):
    return rc.FastMCD(contamination=0.08, quality="balanced", random_state=0).fit(X)


def empirical_covariance(X):
    return np.cov(X, rowvar=False) + 1e-8 * np.eye(X.shape[1])


def robust_rbf_similarity(x, y, precision, length_scale=1.0):
    delta = x - y
    d2 = float(delta @ precision @ delta)
    return float(np.exp(-0.5 * d2 / (length_scale**2)))


def main():
    rng = np.random.default_rng(7)

    print("SPD geometry ML use cases")
    print("=========================")

    # ------------------------------------------------------------------
    # 1. Regime/drift detection with distances between robust scatters.
    # ------------------------------------------------------------------
    baseline = make_window(rng, regime="baseline", contamination=0.08)
    S_ref = robust_scatter(baseline).covariance_

    windows = [
        ("baseline_clean", make_window(rng, regime="baseline", contamination=0.00)),
        ("baseline_contaminated", make_window(rng, regime="baseline", contamination=0.08)),
        ("shifted_clean", make_window(rng, regime="shifted", contamination=0.00)),
        ("shifted_contaminated", make_window(rng, regime="shifted", contamination=0.08)),
    ]

    print()
    print("1) Robust covariance drift monitoring")
    print("window                  affine_distance_to_baseline")
    for name, X in windows:
        S = robust_scatter(X).covariance_
        d = rcg.affine_invariant_distance(S_ref, S)
        print(f"{name:24s} {d:10.4f}")

    # ------------------------------------------------------------------
    # 2. Empirical vs robust scatter under contaminated inputs.
    # ------------------------------------------------------------------
    X_contaminated = make_window(rng, regime="baseline", contamination=0.10)
    S_emp = empirical_covariance(X_contaminated)
    S_rob = robust_scatter(X_contaminated).covariance_

    print()
    print("2) Empirical vs robust scatter under contamination")
    print("empirical covariance diag:", np.round(np.diag(S_emp), 4))
    print("robust covariance diag:   ", np.round(np.diag(S_rob), 4))
    print(
        "affine distance empirical-vs-robust:",
        round(rcg.affine_invariant_distance(S_emp, S_rob), 4),
    )

    # ------------------------------------------------------------------
    # 3. Estimator stability under contamination.
    # ------------------------------------------------------------------
    X_clean = make_window(rng, regime="baseline", contamination=0.00)
    X_corrupted = contaminate_same_sample(rng, X_clean, contamination=0.10)

    S_emp_clean = empirical_covariance(X_clean)
    S_emp_corrupted = empirical_covariance(X_corrupted)

    S_rob_clean = robust_scatter(X_clean).covariance_
    S_rob_corrupted = robust_scatter(X_corrupted).covariance_

    d_emp = rcg.affine_invariant_distance(S_emp_clean, S_emp_corrupted)
    d_rob = rcg.affine_invariant_distance(S_rob_clean, S_rob_corrupted)

    print()
    print("3) Estimator stability under the same contamination")
    print("affine distance clean->corrupted empirical covariance:", round(d_emp, 4))
    print("affine distance clean->corrupted robust covariance:   ", round(d_rob, 4))

    # ------------------------------------------------------------------
    # 4. Robust similarity induced by robust scatter.
    # ------------------------------------------------------------------
    robust_fit = robust_scatter(X_contaminated)
    center = robust_fit.location_
    precision = robust_fit.precision_

    x_near = center + np.array([0.1, 0.1])
    x_far_leverage = center + np.array([0.1, 3.5])

    sim_near = robust_rbf_similarity(center, x_near, precision, length_scale=1.0)
    sim_far = robust_rbf_similarity(center, x_far_leverage, precision, length_scale=1.0)

    print()
    print("4) Robust similarity from robust scatter")
    print("similarity(center, nearby point):        ", f"{sim_near:.4f}")
    print("similarity(center, leverage-like point): ", f"{sim_far:.3e}")

    # ------------------------------------------------------------------
    # 5. Regularization path diagnostics on the SPD cone.
    # ------------------------------------------------------------------
    X_path = make_window(rng, regime="baseline", contamination=0.08)
    X_path = X_path - X_path.mean(axis=0)

    S_tyler = rc.TylerShape(assume_centered=True).fit(X_path).covariance_

    print()
    print("5) Regularized Tyler path geometry")
    print("alpha   dist_to_tyler   condition_number   tyler_residual")
    for alpha in [0.02, 0.05, 0.15, 0.40]:
        est = rc.RegularizedTyler(alpha=alpha, assume_centered=True).fit(X_path)
        S_alpha = est.covariance_
        d_alpha = rcg.affine_invariant_distance(S_tyler, S_alpha)
        cond_alpha = np.linalg.cond(S_alpha)
        resid_alpha = rcg.tyler_fixed_point_residual(S_alpha, X_path)
        print(f"{alpha:5.2f} {d_alpha:14.4f} {cond_alpha:18.3f} {resid_alpha:16.3e}")


    # ------------------------------------------------------------------
    # 6. Robust whitening / preprocessing stability.
    # ------------------------------------------------------------------
    X_clean = make_window(rng, regime="baseline", contamination=0.00)
    X_contaminated = contaminate_same_sample(rng, X_clean, contamination=0.10)

    S_emp_contaminated = empirical_covariance(X_contaminated)
    S_rob_contaminated = robust_scatter(X_contaminated).covariance_

    W_emp = rcg.spd_power(S_emp_contaminated, -0.5)
    W_rob = rcg.spd_power(S_rob_contaminated, -0.5)

    X_clean_emp_white = X_clean @ W_emp.T
    X_clean_rob_white = X_clean @ W_rob.T

    S_clean_emp_white = empirical_covariance(X_clean_emp_white)
    S_clean_rob_white = empirical_covariance(X_clean_rob_white)

    I = np.eye(X_clean.shape[1])
    d_emp_white = rcg.affine_invariant_distance(I, S_clean_emp_white)
    d_rob_white = rcg.affine_invariant_distance(I, S_clean_rob_white)

    print()
    print("6) Robust whitening for preprocessing")
    print("distance of clean whitened covariance to identity")
    print("using empirical covariance from contaminated data:", round(d_emp_white, 4))
    print("using robust covariance from contaminated data:   ", round(d_rob_white, 4))

    # ------------------------------------------------------------------
    # 7. Robust geometry for nearest-neighbor / embedding retrieval.
    # ------------------------------------------------------------------
    X_ref = make_window(rng, regime="baseline", contamination=0.08)
    fit_ref = robust_scatter(X_ref)
    center = fit_ref.location_
    precision = fit_ref.precision_

    query = center + np.array([0.0, 0.0])
    true_neighbor = center + np.array([0.15, 0.08])
    leverage_neighbor = center + np.array([0.05, 2.8])

    euclidean_true = float(np.linalg.norm(query - true_neighbor))
    euclidean_leverage = float(np.linalg.norm(query - leverage_neighbor))

    robust_true = float(
        np.sqrt((query - true_neighbor) @ precision @ (query - true_neighbor))
    )
    robust_leverage = float(
        np.sqrt((query - leverage_neighbor) @ precision @ (query - leverage_neighbor))
    )

    print()
    print("7) Robust geometry for nearest-neighbor ranking")
    print("candidate             euclidean_distance   robust_distance")
    print(f"true nearby point       {euclidean_true:14.4f} {robust_true:17.4f}")
    print(f"leverage-like point     {euclidean_leverage:14.4f} {robust_leverage:17.4f}")

if __name__ == "__main__":
    main()
