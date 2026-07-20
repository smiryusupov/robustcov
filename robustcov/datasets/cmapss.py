# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""NASA C-MAPSS turbofan degradation dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import numpy as np

from ._external import (
    ArchiveSource,
    DatasetIntegrityError,
    ExternalDatasetInfo,
    extract_nested_archives_until,
    find_file,
    prepare_external_dataset,
)


CMAPSS_INFO = ExternalDatasetInfo(
    name="NASA C-MAPSS Turbofan Engine Degradation Simulation",
    slug="cmapss",
    homepage="https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
    citation=(
        "A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation "
        "Data Set, NASA Ames Prognostics Data Repository, NASA Ames Research Center."
    ),
    license_name="License not specified on the NASA Open Data Portal record",
    license_url=None,
    terms_note=(
        "Review the current NASA dataset page and applicable U.S. Government data terms. "
        "robustcov downloads only after an explicit request and never redistributes the archive."
    ),
    sources=(
        ArchiveSource(
            url=(
                "https://phm-datasets.s3.amazonaws.com/NASA/"
                "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
            ),
            checksum="a83e8f128c59fc5614a4ca2e42a2e30c",
            algorithm="md5",
            label="NASA Prognostics Center of Excellence",
        ),
        ArchiveSource(
            url="https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
            checksum="79a22f36e80606c69d0e9e4da5bb2b7a",
            algorithm="md5",
            label="NASA Open Data legacy resource",
        ),
    ),
)

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_NAMES = ("operating_setting_1", "operating_setting_2", "operating_setting_3")
SENSOR_NAMES = tuple(f"sensor_{index}" for index in range(1, 22))
COLUMN_NAMES = ("unit", "cycle", *SETTING_NAMES, *SENSOR_NAMES)


@dataclass(frozen=True)
class CMapssSplit:
    """One C-MAPSS train or test split."""

    unit: np.ndarray
    cycle: np.ndarray
    settings: np.ndarray
    sensors: np.ndarray

    @property
    def X(self) -> np.ndarray:
        """Return operational settings followed by sensor measurements."""

        return np.column_stack((self.settings, self.sensors))

    @property
    def n_units(self) -> int:
        return int(np.unique(self.unit).size)


@dataclass(frozen=True)
class CMapssDataset:
    """Parsed C-MAPSS subset with train, test, and test RUL values."""

    subset: str
    train: CMapssSplit
    test: CMapssSplit
    test_rul: np.ndarray
    setting_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    data_dir: Path
    archive_path: Path
    archive_sha256: str
    info: ExternalDatasetInfo


def _load_table(path: Path) -> CMapssSplit:
    try:
        values = np.loadtxt(path, dtype=float)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"could not parse C-MAPSS table {path}") from exc
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 26:
        raise DatasetIntegrityError(
            f"{path.name} has shape {values.shape}; expected 26 columns"
        )
    if not np.all(np.isfinite(values)):
        raise DatasetIntegrityError(f"{path.name} contains non-finite values")
    unit = values[:, 0].astype(np.int32)
    cycle = values[:, 1].astype(np.int32)
    if np.any(unit <= 0) or np.any(cycle <= 0):
        raise DatasetIntegrityError(f"{path.name} contains non-positive unit or cycle identifiers")
    return CMapssSplit(
        unit=unit,
        cycle=cycle,
        settings=values[:, 2:5].astype(float, copy=False),
        sensors=values[:, 5:26].astype(float, copy=False),
    )


def _load_rul(path: Path, expected_units: int) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=float)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"could not parse C-MAPSS RUL file {path}") from exc
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != expected_units:
        raise DatasetIntegrityError(
            f"{path.name} contains {values.size} RUL values; expected {expected_units}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise DatasetIntegrityError(f"{path.name} contains invalid RUL values")
    return values


def fetch_cmapss(
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD002",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    download: bool = False,
    archive_path: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
) -> CMapssDataset:
    """Load a NASA C-MAPSS subset from a user cache.

    Parameters
    ----------
    subset : {"FD001", "FD002", "FD003", "FD004"}, default="FD002"
        NASA dataset subset. FD002 and FD004 contain six operating conditions.
    cache_dir : path-like, optional
        External-data cache root. See :func:`robustcov.datasets.get_data_home`.
    download : bool, default=False
        Download from NASA when no archive is cached. Network access is never
        used unless this is explicitly true.
    archive_path : path-like, optional
        Existing manually downloaded ZIP archive. It is copied into the cache,
        fingerprinted, and safely extracted.
    timeout : float, default=120
        Network timeout in seconds when downloading.
    """

    subset = str(subset).upper()
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    prepared = prepare_external_dataset(
        CMAPSS_INFO,
        archive_filename="cmapss_turbofan.zip",
        cache_dir=cache_dir,
        download=download,
        archive_path=archive_path,
        timeout=timeout,
    )

    required = (
        f"train_{subset}.txt",
        f"test_{subset}.txt",
        f"RUL_{subset}.txt",
    )
    extract_nested_archives_until(prepared.extracted_dir, required, max_depth=2)
    paths = {name: find_file(prepared.extracted_dir, name) for name in required}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise DatasetIntegrityError(
            f"C-MAPSS archive is missing required files for {subset}: {missing}"
        )

    train = _load_table(paths[required[0]] or Path())
    test = _load_table(paths[required[1]] or Path())
    test_rul = _load_rul(paths[required[2]] or Path(), expected_units=test.n_units)
    if train.n_units <= 0 or test.n_units <= 0:
        raise DatasetIntegrityError("C-MAPSS split contains no engine trajectories")
    return CMapssDataset(
        subset=subset,
        train=train,
        test=test,
        test_rul=test_rul,
        setting_names=SETTING_NAMES,
        sensor_names=SENSOR_NAMES,
        data_dir=prepared.extracted_dir,
        archive_path=prepared.archive_path,
        archive_sha256=prepared.archive_sha256,
        info=CMAPSS_INFO,
    )


__all__ = [
    "CMAPSS_INFO",
    "COLUMN_NAMES",
    "SETTING_NAMES",
    "SENSOR_NAMES",
    "SUBSETS",
    "CMapssDataset",
    "CMapssSplit",
    "fetch_cmapss",
]
