# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Helpers for optional external-data and Kaggle-style examples."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np


def top_k_mask(scores, k=None, contamination=None):
    """Return a boolean mask for the largest anomaly scores.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Larger values are treated as more anomalous.
    k : int, optional
        Number of observations to mark as anomalous.
    contamination : float, optional
        Fraction of observations to mark as anomalous when ``k`` is omitted.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if k is None:
        if contamination is None:
            raise ValueError("Provide either k or contamination")
        k = int(np.ceil(float(contamination) * n))
    k = max(0, min(int(k), n))
    mask = np.zeros(n, dtype=bool)
    if k:
        idx = np.argpartition(s, n - k)[n - k:]
        mask[idx] = True
    return mask


def scores_to_submission(ids, scores, output_path, id_column="id", score_column="score", higher_is_anomaly=True):
    """Write a Kaggle-style two-column submission from anomaly scores.

    This helper intentionally does not assume a specific competition target name.
    Some competitions require probabilities, others require labels or scores.  Use
    ``score_column`` to match the expected submission column.
    """
    ids = list(ids)
    scores = np.asarray(scores, dtype=float).ravel()
    if len(ids) != scores.size:
        raise ValueError("ids and scores must have the same length")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vals = scores if higher_is_anomaly else -scores
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    if hi > lo:
        vals = (vals - lo) / (hi - lo)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([id_column, score_column])
        w.writerows(zip(ids, vals))
    return out


__all__ = ["top_k_mask", "scores_to_submission"]
