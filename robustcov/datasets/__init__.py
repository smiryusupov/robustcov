# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Optional external dataset loaders.

The loaders never access the network during import.  Downloads require an
explicit ``download=True`` argument or the ``python -m robustcov.datasets fetch``
command, and all raw files are stored in a user cache outside the repository.
"""

from ._external import (
    ArchiveSource,
    DatasetDownloadError,
    DatasetIntegrityError,
    DatasetNotFoundError,
    ExternalDatasetInfo,
    get_data_home,
)
from .cmapss import CMapssDataset, CMapssSplit, CMAPSS_INFO, fetch_cmapss
from .gas_sensor_drift import (
    GasSensorDriftDataset,
    GAS_SENSOR_DRIFT_INFO,
    fetch_gas_sensor_drift,
)


DATASET_REGISTRY = {
    "gas_sensor_drift": GAS_SENSOR_DRIFT_INFO,
    "cmapss": CMAPSS_INFO,
}


__all__ = [
    "ArchiveSource",
    "CMapssDataset",
    "CMapssSplit",
    "CMAPSS_INFO",
    "DATASET_REGISTRY",
    "DatasetDownloadError",
    "DatasetIntegrityError",
    "DatasetNotFoundError",
    "ExternalDatasetInfo",
    "GAS_SENSOR_DRIFT_INFO",
    "GasSensorDriftDataset",
    "fetch_cmapss",
    "fetch_gas_sensor_drift",
    "get_data_home",
]
