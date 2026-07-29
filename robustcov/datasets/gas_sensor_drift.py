# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""UCI Gas Sensor Array Drift at Different Concentrations loader."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from ._external import (
    ArchiveSource,
    DatasetIntegrityError,
    ExternalDatasetInfo,
    find_file,
    prepare_external_dataset,
)


GAS_SENSOR_DRIFT_INFO = ExternalDatasetInfo(
    name="UCI Gas Sensor Array Drift at Different Concentrations",
    slug="gas_sensor_drift",
    homepage=(
        "https://archive.ics.uci.edu/dataset/270/"
        "gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations"
    ),
    citation=(
        "Vergara, A. (2012). Gas Sensor Array Drift at Different Concentrations "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MK6M"
    ),
    license_name="CC BY 4.0 on the current UCI dataset page",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    terms_note=(
        "The current UCI metadata lists CC BY 4.0, while older descriptive text on the "
        "same record contains a research-only statement. Review the current source page "
        "and your intended use before downloading. robustcov does not redistribute the data."
    ),
    sources=(
        ArchiveSource(
            url=(
                "https://archive.ics.uci.edu/static/public/270/"
                "gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip"
            ),
            label="UCI Machine Learning Repository",
        ),
    ),
)

GAS_NAMES = (
    "Ethanol",
    "Ethylene",
    "Ammonia",
    "Acetaldehyde",
    "Acetone",
    "Toluene",
)

FEATURE_NAMES = tuple(
    f"{feature}_{sensor}"
    for sensor in range(1, 17)
    for feature in (
        "DR",
        "abs_DR",
        "EMAi_0.001",
        "EMAi_0.01",
        "EMAi_0.1",
        "EMAd_0.001",
        "EMAd_0.01",
        "EMAd_0.1",
    )
)

_BATCH_PATTERN = re.compile(r"batch(\d+)\.dat$", re.IGNORECASE)


@dataclass(frozen=True)
class GasSensorDriftDataset:
    """Parsed gas-sensor drift measurements and temporal batch labels."""

    X: np.ndarray
    gas: np.ndarray
    concentration: np.ndarray
    batch: np.ndarray
    gas_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    data_dir: Path
    archive_path: Path
    archive_sha256: str
    info: ExternalDatasetInfo

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.gas_names


def _parse_line(line: str, *, filename: str, line_number: int) -> tuple[int, float, np.ndarray]:
    tokens = line.split()
    if len(tokens) != 129:
        raise DatasetIntegrityError(
            f"{filename}:{line_number} has {len(tokens)} fields; expected class/concentration "
            "plus 128 libsvm-style features"
        )
    try:
        gas_text, concentration_text = tokens[0].split(";", maxsplit=1)
        gas = int(gas_text)
        concentration = float(concentration_text)
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(
            f"{filename}:{line_number} has an invalid class/concentration field"
        ) from exc
    if not 1 <= gas <= len(GAS_NAMES):
        raise DatasetIntegrityError(f"{filename}:{line_number} has invalid gas code {gas}")

    values = np.full(128, np.nan, dtype=float)
    for token in tokens[1:]:
        try:
            index_text, value_text = token.split(":", maxsplit=1)
            index = int(index_text) - 1
            value = float(value_text)
        except (TypeError, ValueError) as exc:
            raise DatasetIntegrityError(
                f"{filename}:{line_number} has invalid feature token {token!r}"
            ) from exc
        if not 0 <= index < 128:
            raise DatasetIntegrityError(
                f"{filename}:{line_number} has feature index {index + 1}, expected 1..128"
            )
        if np.isfinite(values[index]):
            raise DatasetIntegrityError(
                f"{filename}:{line_number} repeats feature index {index + 1}"
            )
        values[index] = value
    if not np.all(np.isfinite(values)):
        raise DatasetIntegrityError(f"{filename}:{line_number} has missing or non-finite features")
    return gas, concentration, values


def _find_batches(extracted_dir: Path) -> list[tuple[int, Path]]:
    batches: list[tuple[int, Path]] = []
    for path in extracted_dir.rglob("batch*.dat"):
        match = _BATCH_PATTERN.search(path.name)
        if match and path.is_file():
            batches.append((int(match.group(1)), path))
    batches.sort(key=lambda item: item[0])
    identifiers = [batch for batch, _ in batches]
    if identifiers != list(range(1, 11)):
        raise DatasetIntegrityError(
            f"expected batch1.dat through batch10.dat, found batch identifiers {identifiers}"
        )
    return batches


def _normalize_selection(values: Iterable[int] | None, *, minimum: int, maximum: int, name: str) -> set[int] | None:
    if values is None:
        return None
    selected = {int(value) for value in values}
    if not selected:
        raise ValueError(f"{name} must not be empty")
    invalid = sorted(value for value in selected if not minimum <= value <= maximum)
    if invalid:
        raise ValueError(f"invalid {name}: {invalid}; expected values in {minimum}..{maximum}")
    return selected


def fetch_gas_sensor_drift(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    download: bool = False,
    archive_path: str | os.PathLike[str] | None = None,
    batches: Iterable[int] | None = None,
    gases: Iterable[int] | None = None,
    timeout: float = 120.0,
) -> GasSensorDriftDataset:
    """Load the UCI gas-sensor drift dataset from a user cache.

    Parameters
    ----------
    cache_dir : path-like, optional
        External-data cache root. See :func:`robustcov.datasets.get_data_home`.
    download : bool, default=False
        Download from UCI when the archive is not cached. No network access
        occurs unless this is explicitly true.
    archive_path : path-like, optional
        Existing manually downloaded ZIP archive. It is copied into the cache,
        fingerprinted, and safely extracted.
    batches : iterable of int, optional
        Temporal batches to return, using identifiers 1 through 10.
    gases : iterable of int, optional
        Gas codes to return, using identifiers 1 through 6.
    timeout : float, default=120
        Network timeout in seconds when downloading.
    """

    prepared = prepare_external_dataset(
        GAS_SENSOR_DRIFT_INFO,
        archive_filename="gas_sensor_drift_uci_270.zip",
        cache_dir=cache_dir,
        download=download,
        archive_path=archive_path,
        timeout=timeout,
    )
    selected_batches = _normalize_selection(batches, minimum=1, maximum=10, name="batches")
    selected_gases = _normalize_selection(gases, minimum=1, maximum=6, name="gases")

    X_rows: list[np.ndarray] = []
    gas_rows: list[int] = []
    concentration_rows: list[float] = []
    batch_rows: list[int] = []
    for batch_id, path in _find_batches(prepared.extracted_dir):
        if selected_batches is not None and batch_id not in selected_batches:
            continue
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                gas, concentration, values = _parse_line(
                    line, filename=path.name, line_number=line_number
                )
                if selected_gases is not None and gas not in selected_gases:
                    continue
                X_rows.append(values)
                gas_rows.append(gas)
                concentration_rows.append(concentration)
                batch_rows.append(batch_id)

    if not X_rows:
        raise DatasetIntegrityError("the requested gas-sensor subset contains no rows")
    X = np.vstack(X_rows)
    gas = np.asarray(gas_rows, dtype=np.int16)
    concentration = np.asarray(concentration_rows, dtype=float)
    batch = np.asarray(batch_rows, dtype=np.int16)
    if selected_batches is None and selected_gases is None and X.shape != (13_910, 128):
        raise DatasetIntegrityError(
            f"full gas-sensor archive has shape {X.shape}; expected (13910, 128)"
        )
    return GasSensorDriftDataset(
        X=X,
        gas=gas,
        concentration=concentration,
        batch=batch,
        gas_names=GAS_NAMES,
        feature_names=FEATURE_NAMES,
        data_dir=prepared.extracted_dir,
        archive_path=prepared.archive_path,
        archive_sha256=prepared.archive_sha256,
        info=GAS_SENSOR_DRIFT_INFO,
    )


__all__ = [
    "FEATURE_NAMES",
    "GAS_NAMES",
    "GAS_SENSOR_DRIFT_INFO",
    "GasSensorDriftDataset",
    "fetch_gas_sensor_drift",
]
