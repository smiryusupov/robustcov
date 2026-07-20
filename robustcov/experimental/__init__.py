# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Experimental estimators with APIs or defaults that may change."""

from .distributionally_robust_pca import (
    DistributionallyRobustPCA,
    WassersteinRobustPCA,
)

from ..provenance import attach_method_provenance

__all__ = ["DistributionallyRobustPCA", "WassersteinRobustPCA"]

attach_method_provenance(globals())
