"""Plot ML use-case diagnostics for SPD geometry utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import robustcov.geometry as rcg
from spd_geometry_ml_use_cases import (
    contaminate_same_sample,
    empirical_covariance,
    make_window,
    robust_scatter,
)


def main():
    rng = np.random.default_rng(7)
    outdir = Path("results/examples/spd_geometry")
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: covariance drift monitoring.
    # ------------------------------------------------------------------
    baseline = make_window(rng, regime="baseline", contamination=0.08)
    S_ref = robust_scatter(baseline).covariance_

    windows = [
        ("baseline\nclean", make_window(rng, regime="baseline", contamination=0.00)),
        ("baseline\ncontaminated", make_window(rng, regime="baseline", contamination=0.08)),
        ("shifted\nclean", make_window(rng, regime="shifted", contamination=0.00)),
        ("shifted\ncontaminated", make_window(rng, regime="shifted", contamination=0.08)),
    ]

    labels = []
    drift_scores = []
    for name, X in windows:
        S = robust_scatter(X).covariance_
        labels.append(name)
        drift_scores.append(rcg.affine_invariant_distance(S_ref, S))

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(labels, drift_scores)
    ax.set_ylabel("Affine-invariant distance to baseline")
    ax.set_title("Robust covariance drift monitoring")
    ax.set_ylim(0.0, max(drift_scores) * 1.15)
    fig.tight_layout()
    fig.savefig(outdir / "spd_geometry_drift_monitoring.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: where robust geometry helps under contamination.
    # ------------------------------------------------------------------
    X_clean = make_window(rng, regime="baseline", contamination=0.00)
    X_corrupted = contaminate_same_sample(rng, X_clean, contamination=0.10)

    S_emp_clean = empirical_covariance(X_clean)
    S_emp_corrupted = empirical_covariance(X_corrupted)
    S_rob_clean = robust_scatter(X_clean).covariance_
    S_rob_corrupted = robust_scatter(X_corrupted).covariance_

    d_emp_stability = rcg.affine_invariant_distance(S_emp_clean, S_emp_corrupted)
    d_rob_stability = rcg.affine_invariant_distance(S_rob_clean, S_rob_corrupted)

    W_emp = rcg.spd_power(S_emp_corrupted, -0.5)
    W_rob = rcg.spd_power(S_rob_corrupted, -0.5)

    X_clean_emp_white = X_clean @ W_emp.T
    X_clean_rob_white = X_clean @ W_rob.T

    I = np.eye(X_clean.shape[1])
    d_emp_white = rcg.affine_invariant_distance(
        I, empirical_covariance(X_clean_emp_white)
    )
    d_rob_white = rcg.affine_invariant_distance(
        I, empirical_covariance(X_clean_rob_white)
    )

    groups = ["Estimator\nmovement", "Whitened clean\ncovariance error"]
    empirical_values = [d_emp_stability, d_emp_white]
    robust_values = [d_rob_stability, d_rob_white]

    x = np.arange(len(groups))
    width = 0.35

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, empirical_values, width, label="empirical covariance")
    ax.bar(x + width / 2, robust_values, width, label="robust covariance")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Affine-invariant distance")
    ax.set_title("Robust geometry is more stable under contamination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "spd_geometry_stability_whitening.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved plots to", outdir)


if __name__ == "__main__":
    main()
