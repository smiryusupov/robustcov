# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._utils import check_array, median_impute


@dataclass
class RobustMedianImputer:
    """Small sklearn-like column-median imputer for examples and pipelines."""

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=True)
        _, values = median_impute(X)
        self.statistics_ = values
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "statistics_"):
            raise RuntimeError("RobustMedianImputer is not fitted")
        X = check_array(X, allow_nan=True)
        return median_impute(X, self.statistics_)[0]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
