"""Example: SPD geometry diagnostics for robust scatter estimates.

This example shows how to use ``robustcov.geometry`` to compare robust scatter
matrices, normalize shape matrices, walk along an SPD geodesic, and inspect a
Tyler fixed-point residual.

The geometry module is utility-focused: it helps diagnose and compare covariance
or scatter matrices, but it does not introduce a new estimator family.
"""

from __future__ import annotations

import numpy as np

import robustcov as rc
import robustcov.geometry as rcg


def make_heavy_tailed_data(seed: int, n_samples: int = 120):
    rng = np.random.default_rng(seed)

    # Correlated 2D base distribution.
    A = np.array([[1.0, 0.7], [0.0, 0.5]])
    Z = rng.normal(size=(n_samples, 2)) @ A.T

    # Random radial scales create heavy tails while keeping the same directions.
    scales = np.sqrt(rng.gamma(shape=1.5, scale=1.0, size=n_samples))
    return Z / scales[:, None]


def main():
    X1 = make_heavy_tailed_data(seed=0)
    X2 = make_heavy_tailed_data(seed=1)

    # Add a mild regime change to X2.
    X2 = X2 @ np.array([[1.2, -0.25], [0.0, 0.8]]).T

    est1 = rc.TylerShape(assume_centered=True).fit(X1)
    est2 = rc.RegularizedTyler(alpha=0.15, assume_centered=True).fit(X2)

    S1 = est1.covariance_
    S2 = est2.covariance_

    print("SPD geometry diagnostics")
    print("------------------------")

    print("trace(S1) before normalization:", round(float(np.trace(S1)), 6))
    print("trace(trace_normalize(S1)):    ", round(float(np.trace(rcg.trace_normalize(S1))), 6))
    print("det(det_normalize(S1)):        ", round(float(np.linalg.det(rcg.det_normalize(S1))), 6))
    print()

    print("affine-invariant distance(S1, S2):", round(rcg.affine_invariant_distance(S1, S2), 6))
    print("log-Euclidean distance(S1, S2):    ", round(rcg.logeuclidean_distance(S1, S2), 6))
    print()

    print("Tyler fixed-point residuals")
    print("S1 on X1:", f"{rcg.tyler_fixed_point_residual(S1, X1):.3e}")
    print("S2 on X2:", f"{rcg.tyler_fixed_point_residual(S2, X2):.3e}")
    print()

    print("Geodesic from S1 to S2")
    print("t    distance_from_S1    residual_on_X1")
    for t in np.linspace(0.0, 1.0, 5):
        S_t = rcg.spd_geodesic(S1, S2, float(t))
        d_t = rcg.affine_invariant_distance(S1, S_t)
        r_t = rcg.tyler_fixed_point_residual(S_t, X1)
        print(f"{t:0.2f} {d_t:17.6f} {r_t:17.3e}")


if __name__ == "__main__":
    main()
