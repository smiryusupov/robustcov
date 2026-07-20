# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from robustcov.datasets import (
    DatasetIntegrityError,
    DatasetNotFoundError,
    fetch_cmapss,
    fetch_gas_sensor_drift,
    get_data_home,
)
from robustcov.datasets._external import safe_extract_zip


def _gas_line(gas: int, concentration: float, offset: float) -> str:
    features = " ".join(f"{index}:{offset + index / 100.0:.6f}" for index in range(1, 129))
    return f"{gas};{concentration:.6f} {features}\n"


def _make_gas_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for batch in range(1, 11):
            content = _gas_line((batch - 1) % 6 + 1, 10.0 + batch, float(batch))
            archive.writestr(f"dataset/batch{batch}.dat", content)


def _cmapss_row(unit: int, cycle: int, offset: float) -> str:
    values = [unit, cycle, 0.1, 0.2, 0.3, *[offset + index / 10.0 for index in range(1, 22)]]
    return " ".join(str(value) for value in values) + "\n"


def _make_cmapss_archive(path: Path, *, nested: bool = True) -> None:
    inner = path.with_name("inner.zip")
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        train = "".join(
            _cmapss_row(unit, cycle, unit + cycle / 100.0)
            for unit in (1, 2)
            for cycle in range(1, 6)
        )
        test = "".join(
            _cmapss_row(unit, cycle, 2 * unit + cycle / 100.0)
            for unit in (1, 2)
            for cycle in range(1, 4)
        )
        archive.writestr("CMAPSSData/train_FD002.txt", train)
        archive.writestr("CMAPSSData/test_FD002.txt", test)
        archive.writestr("CMAPSSData/RUL_FD002.txt", "10\n20\n")
    if nested:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(inner, arcname="CMAPSSData.zip")
        inner.unlink()
    else:
        path.write_bytes(inner.read_bytes())
        inner.unlink()


def test_data_home_precedence(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    assert get_data_home(explicit) == explicit.resolve()

    configured = tmp_path / "configured"
    monkeypatch.setenv("ROBUSTCOV_DATA_DIR", str(configured))
    assert get_data_home() == configured.resolve()

    monkeypatch.delenv("ROBUSTCOV_DATA_DIR")
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert get_data_home() == (xdg / "robustcov").resolve()


def test_fetchers_do_not_download_by_default(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="download=True"):
        fetch_gas_sensor_drift(cache_dir=tmp_path)
    with pytest.raises(DatasetNotFoundError, match="download=True"):
        fetch_cmapss(cache_dir=tmp_path)


def test_gas_sensor_loader_parses_local_archive_and_pins_fingerprint(tmp_path):
    archive = tmp_path / "gas.zip"
    _make_gas_archive(archive)
    dataset = fetch_gas_sensor_drift(
        cache_dir=tmp_path / "cache",
        archive_path=archive,
        batches=(1, 2),
    )
    assert dataset.X.shape == (2, 128)
    np.testing.assert_array_equal(dataset.batch, [1, 2])
    np.testing.assert_array_equal(dataset.gas, [1, 2])
    assert dataset.concentration.tolist() == [11.0, 12.0]
    assert len(dataset.archive_sha256) == 64
    sidecars = list((tmp_path / "cache" / "gas_sensor_drift" / "raw").glob("*.sha256.json"))
    assert len(sidecars) == 1


def test_cmapss_loader_handles_nested_archive(tmp_path):
    archive = tmp_path / "cmapss.zip"
    _make_cmapss_archive(archive)
    dataset = fetch_cmapss(
        "FD002",
        cache_dir=tmp_path / "cache",
        archive_path=archive,
    )
    assert dataset.train.sensors.shape == (10, 21)
    assert dataset.train.settings.shape == (10, 3)
    assert dataset.test.sensors.shape == (6, 21)
    np.testing.assert_array_equal(dataset.test_rul, [10.0, 20.0])
    assert dataset.train.n_units == 2
    assert dataset.test.n_units == 2


def test_safe_zip_extraction_rejects_traversal_and_symlinks(tmp_path):
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(DatasetIntegrityError, match="unsafe archive member"):
        safe_extract_zip(traversal, tmp_path / "out1")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(DatasetIntegrityError, match="Symbolic|symbolic"):
        safe_extract_zip(symlink, tmp_path / "out2")
