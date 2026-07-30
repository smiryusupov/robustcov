# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Temporary direct-extension checks for native input validation."""

import numpy as np
import pytest

from robustcov._native import native_available, require_native


def test_native_boundaries_reject_malformed_inputs():
    """Representative bad inputs fail before native indexing or allocation."""
    if not native_available():
        pytest.skip("native extension unavailable")
    cpp = require_native("native boundary validation")

    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        cpp.fit_tyler(X, max_iter=0)
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        cpp.fit_tyler(X, regularization=0.1, tol=np.nan)
    with pytest.raises(ValueError, match="support_fraction"):
        cpp.fit_fast_mcd(X, support_fraction=np.nan)

    location = np.zeros(3, dtype=np.float64)
    location[1] = np.inf
    with pytest.raises(ValueError, match="location contains NaN or infinity"):
        cpp.mahalanobis2_batch(X, location, np.eye(3))

    with pytest.raises(ValueError, match="X rows must be at least 1"):
        cpp.matrix_mahalanobis2_batch(
            np.empty((1, 0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.eye(2),
        )

    matrix_samples = np.ones((2, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="row component rank must be at least 1"):
        cpp.weighted_tucker_scores_2d(
            matrix_samples,
            np.ones_like(matrix_samples),
            np.zeros((2, 2), dtype=np.float64),
            np.empty((2, 0), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
        )

    matrices = np.eye(2, dtype=np.float64)[None, :, :]
    matrices[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="matrices contain NaN or infinity"):
        cpp.joint_diagonalize_symmetric(matrices)
