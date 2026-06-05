"""Use case: network traffic anomaly simulation.

Features imitate flow duration, packet count, byte count, byte/packet ratio,
failed connection indicators, and entropy-like summaries. Robust covariance
provides a simple multivariate baseline for unusual flows.

Run:
    python examples/use_case_network_traffic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import robustcov as rc


def precision_recall(pred, labels):
    pred = np.asarray(pred, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    return (np.sum(pred & labels) / max(1, np.sum(pred)),
            np.sum(pred & labels) / max(1, np.sum(labels)))


if __name__ == "__main__":
    rng = np.random.default_rng(22)
    n, p = 1500, 7
    X = rng.normal(size=(n, p))
    # induce normal correlation among flow features
    X[:, 1] = 0.8 * X[:, 0] + 0.5 * rng.normal(size=n)
    X[:, 2] = 0.7 * X[:, 1] + 0.6 * rng.normal(size=n)
    labels = np.zeros(n, dtype=int)

    # DDoS-like short, high packet-count flows
    ddos = rng.choice(n, size=45, replace=False)
    labels[ddos] = 1
    X[ddos, 0] -= 3.0
    X[ddos, 1] += 5.0
    X[ddos, 2] += 4.0

    # exfiltration-like long/high-byte/low-entropy flows
    remaining = np.setdiff1d(np.arange(n), ddos)
    exfil = rng.choice(remaining, size=35, replace=False)
    labels[exfil] = 1
    X[exfil, 0] += 4.0
    X[exfil, 2] += 5.0
    X[exfil, 5] -= 4.0

    est = rc.FastMCD(quality="fast", random_state=3).fit(X)
    det = rc.RobustOutlierDetector(estimator=est, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("network traffic anomaly simulation")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")
    print(f"radial_kurtosis={est.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/network")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_panel(est, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
