# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Experimental estimators with APIs or defaults that may change."""

from .distributionally_robust_pca import (
    DistributionallyRobustPCA,
    WassersteinRobustPCA,
)

from ..online_subspace import OnlineRobustSubspaceTracker, OnlineSubspaceUpdate

from ..provenance import attach_method_provenance

__all__ = [
    "DistributionallyRobustPCA",
    "WassersteinRobustPCA",
    "OnlineRobustSubspaceTracker",
    "OnlineSubspaceUpdate",
]

attach_method_provenance(globals())
