# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_external_snapshot.py"
SPEC = importlib.util.spec_from_file_location("publish_external_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


def _write_cmapss_results(path: Path, *, subset: str = "FD002") -> None:
    path.mkdir(parents=True)
    for name in (
        "risk_over_engine_life.png",
        "alert_rate_by_life.png",
        "late_life_sensor_contributions.png",
    ):
        (path / name).write_bytes(b"\x89PNG\r\n\x1a\nreviewed-test-output")
    (path / "summary.csv").write_text(
        "method,life_bin,alert_rate\nDRO-PCA,0.8-1.0,0.935\n",
        encoding="utf-8",
    )
    (path / "window_scores.csv").write_text("private,row\n1,2\n", encoding="utf-8")
    with (path / "run_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerow(["dataset_subset", subset])
        writer.writerow(["archive_sha256", "abc123"])
        writer.writerow(["dataset_citation", "Example citation"])
        writer.writerow(["dataset_homepage", "https://example.test/dataset"])
        writer.writerow(["cache_dir", "/private/local/cache"])


def test_publish_external_snapshot_copies_only_reviewed_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs/_generated").mkdir(parents=True)
    (root / "docs/external_results").mkdir(parents=True)
    results = tmp_path / "results"
    _write_cmapss_results(results)

    destination = publisher.publish(
        root,
        publisher.PROFILES["cmapss_fd002"],
        results,
        command="python examples_external/cmapss_dro_pca_monitoring.py --subset FD002",
        generated_at="2026-07-20T00:00:00+00:00",
        replace=False,
    )

    assert (destination / "risk_over_engine_life.png").is_file()
    assert (destination / "summary.csv").is_file()
    assert not (destination / "window_scores.csv").exists()
    payload = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["archive_sha256"] == "abc123"
    assert "cache_dir" not in payload["metadata"]
    assert payload["files"]["risk_over_engine_life.png"]["sha256"]
    assert (root / "docs/external_results/cmapss_fd002.rst").is_file()
    manifest = json.loads(
        (root / "docs/_static/external_results/manifest.json").read_text(encoding="utf-8")
    )
    assert [item["slug"] for item in manifest["snapshots"]] == ["cmapss_fd002"]
    generated_toctree = (
        root / "docs/_generated/external_snapshot_toctree.rst"
    ).read_text(encoding="utf-8")
    assert "   external_results/cmapss_fd002" in generated_toctree
    assert "../external_results/cmapss_fd002" not in generated_toctree
    publisher.check(root)


def test_publish_external_snapshot_requires_replace(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    results = tmp_path / "results"
    _write_cmapss_results(results)
    profile = publisher.PROFILES["cmapss_fd002"]
    kwargs = {
        "command": "python protocol.py --subset FD002",
        "generated_at": "2026-07-20T00:00:00+00:00",
        "replace": False,
    }
    publisher.publish(root, profile, results, **kwargs)
    with pytest.raises(publisher.SnapshotError, match="--replace"):
        publisher.publish(root, profile, results, **kwargs)


def test_remove_external_snapshot_regenerates_registry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    results = tmp_path / "results"
    _write_cmapss_results(results)
    publisher.publish(
        root,
        publisher.PROFILES["cmapss_fd002"],
        results,
        command="python protocol.py --subset FD002",
        generated_at="2026-07-20T00:00:00+00:00",
        replace=False,
    )

    assert publisher.remove(root, "cmapss_fd002")
    assert not (root / "docs/_static/external_results/cmapss_fd002").exists()
    assert not (root / "docs/external_results/cmapss_fd002.rst").exists()
    manifest = json.loads(
        (root / "docs/_static/external_results/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["snapshots"] == []
    publisher.check(root)


def test_cmapss_profile_rejects_wrong_subset_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    results = tmp_path / "results"
    _write_cmapss_results(results, subset="FD004")
    with pytest.raises(publisher.SnapshotError, match="dataset_subset"):
        publisher.publish(
            root,
            publisher.PROFILES["cmapss_fd002"],
            results,
            command="python protocol.py --subset FD002",
            generated_at="2026-07-20T00:00:00+00:00",
            replace=False,
        )
