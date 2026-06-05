"""Use case: embedding outlier screening.

This synthetic example imitates sentence/document embeddings. Most points come
from a semantic cluster mixture; a few points represent off-topic or corrupted
embeddings. Robust scatter is used as a quick embedding-quality screen.

Run:
    python examples/use_case_text_embedding_outliers.py
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
    rng = np.random.default_rng(25)
    n, p = 900, 32
    centers = rng.normal(0, 1.0, size=(3, p))
    group = rng.integers(0, 3, size=n)
    X = centers[group] + 0.6 * rng.normal(size=(n, p))
    labels = np.zeros(n, dtype=int)
    idx = rng.choice(n, size=55, replace=False)
    labels[idx] = 1
    X[idx] = rng.normal(0, 5.0, size=(idx.size, p))

    est = rc.AutoRobustScatter(selection="diagnostic").fit(X)
    det = rc.RobustOutlierDetector(estimator=est.estimator_, threshold="empirical", alpha=1 - labels.mean()).fit(X)
    pred = det.labels_ == -1
    precision, recall = precision_recall(pred, labels)

    print("embedding outlier screening")
    print(f"selected={est.best_estimator_name_}")
    print(f"precision={precision:.3f}, recall={recall:.3f}, detected={int(pred.sum())}")

    outdir = Path("results/use_cases/embedding_outliers")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_panel(est.estimator_, labels=labels, output_path=outdir / "distance_panel.png", show=False)
    print("saved diagnostics to", outdir)
