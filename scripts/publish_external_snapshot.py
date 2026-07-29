#!/usr/bin/env python3
# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Publish reviewed external-benchmark outputs as documentation snapshots.

The benchmark itself is run locally or in the manually triggered external-data
workflow.  This script copies only a small allowlist of aggregate outputs into
``docs/_static/external_results`` and writes provenance metadata and RST pages.
It never downloads a dataset and never publishes row-level/window-level output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 2
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TABLE_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".svg", ".csv", ".json"}
PRIVATE_METADATA_KEYS = {"cache_dir", "archive_path", "data_dir"}


@dataclass(frozen=True)
class SnapshotProfile:
    slug: str
    title: str
    protocol: str
    default_results: str
    figures: tuple[tuple[str, str, str], ...]
    tables: tuple[str, ...]
    description: str
    expected_metadata: tuple[tuple[str, str], ...] = ()

    @property
    def required_files(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self.figures) + self.tables + ("run_metadata.csv",)


PROFILES: dict[str, SnapshotProfile] = {
    "cmapss_fd002": SnapshotProfile(
        slug="cmapss_fd002",
        title="NASA C-MAPSS FD002",
        protocol="DRO-PCA degradation monitoring across operating regimes",
        default_results="results/external/cmapss_fd002",
        figures=(
            ("risk_over_engine_life.png", "Risk over engine life", "Rolling residual risk over normalized engine life."),
            ("alert_rate_by_life.png", "Alert rate by life interval", "Alert rates summarized by normalized-life interval."),
            ("late_life_sensor_contributions.png", "Late-life sensor contributions", "Sensors contributing most to late-life residual risk."),
        ),
        tables=("summary.csv",),
        description=(
            "FD002 contains six operating conditions and one fault mode, separating tolerated "
            "regime changes from progressive degradation."
        ),
        expected_metadata=(("dataset_subset", "FD002"),),
    ),
    "cmapss_fd004": SnapshotProfile(
        slug="cmapss_fd004",
        title="NASA C-MAPSS FD004",
        protocol="DRO-PCA degradation monitoring across operating regimes",
        default_results="results/external/cmapss_fd004",
        figures=(
            ("risk_over_engine_life.png", "Risk over engine life", "Rolling residual risk over normalized engine life."),
            ("alert_rate_by_life.png", "Alert rate by life interval", "Alert rates summarized by normalized-life interval."),
            ("late_life_sensor_contributions.png", "Late-life sensor contributions", "Sensors contributing most to late-life residual risk."),
        ),
        tables=("summary.csv",),
        description=(
            "FD004 contains six operating conditions and two fault modes, providing the harder "
            "anticipated-regime-versus-degradation benchmark."
        ),
        expected_metadata=(("dataset_subset", "FD004"),),
    ),
}


class SnapshotError(RuntimeError):
    """Raised when a result directory cannot be safely published."""


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read_metadata(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SnapshotError(f"could not read metadata file {path}") from exc
    if not rows or set(rows[0]) != {"key", "value"}:
        raise SnapshotError(f"{path} must have key,value columns")
    result: dict[str, str] = {}
    for row in rows:
        key = str(row.get("key", "")).strip()
        if not key or key in PRIVATE_METADATA_KEYS:
            continue
        result[key] = str(row.get("value", ""))
    return result


def _read_cmapss_summary(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    required_columns = {"method", "life_bin", "mean_risk", "alert_rate", "n_windows"}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise SnapshotError(
                    f"{path} must contain columns {sorted(required_columns)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SnapshotError(f"could not read C-MAPSS summary {path}") from exc
    if not rows:
        raise SnapshotError(f"C-MAPSS summary is empty: {path}")

    normalized: list[dict[str, object]] = []
    by_method_bin: dict[tuple[str, str], dict[str, object]] = {}
    for raw in rows:
        method = str(raw.get("method", "")).strip()
        life_bin = str(raw.get("life_bin", "")).strip()
        try:
            mean_risk = float(raw["mean_risk"])
            alert_rate = float(raw["alert_rate"])
            n_windows = int(raw["n_windows"])
        except (TypeError, ValueError, KeyError) as exc:
            raise SnapshotError(f"invalid numerical value in {path}: {raw}") from exc
        if method not in {"Empirical PCA", "DRO-PCA"}:
            raise SnapshotError(f"unexpected method in {path}: {method!r}")
        if not math.isfinite(mean_risk) or mean_risk < 0.0:
            raise SnapshotError(f"invalid mean_risk in {path}: {mean_risk!r}")
        if not math.isfinite(alert_rate) or not 0.0 <= alert_rate <= 1.0:
            raise SnapshotError(f"invalid alert_rate in {path}: {alert_rate!r}")
        if n_windows <= 0:
            raise SnapshotError(f"n_windows must be positive in {path}: {n_windows!r}")
        key = (method, life_bin)
        if key in by_method_bin:
            raise SnapshotError(f"duplicate method/life_bin row in {path}: {key}")
        row = {
            "method": method,
            "life_bin": life_bin,
            "mean_risk": mean_risk,
            "alert_rate": alert_rate,
            "n_windows": n_windows,
        }
        normalized.append(row)
        by_method_bin[key] = row

    life_bins = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
    for method in ("Empirical PCA", "DRO-PCA"):
        missing = [life_bin for life_bin in life_bins if (method, life_bin) not in by_method_bin]
        if missing:
            raise SnapshotError(f"{path} is missing {method} life bins: {missing}")

    methods: dict[str, dict[str, object]] = {}
    for method in ("Empirical PCA", "DRO-PCA"):
        early = by_method_bin[(method, "0.0-0.2")]
        late = by_method_bin[(method, "0.8-1.0")]
        ordered = [by_method_bin[(method, life_bin)] for life_bin in life_bins]
        methods[method] = {
            "early_alert_rate": early["alert_rate"],
            "late_alert_rate": late["alert_rate"],
            "alert_rate_change": float(late["alert_rate"]) - float(early["alert_rate"]),
            "early_mean_risk": early["mean_risk"],
            "late_mean_risk": late["mean_risk"],
            "risk_ratio_late_to_early": (
                float(late["mean_risk"]) / float(early["mean_risk"])
                if float(early["mean_risk"]) > 0.0
                else None
            ),
            "total_windows": sum(int(row["n_windows"]) for row in ordered),
            "alert_rates_monotone": all(
                float(left["alert_rate"]) <= float(right["alert_rate"]) + 1e-12
                for left, right in zip(ordered, ordered[1:])
            ),
        }
    return normalized, {"life_bins": list(life_bins), "methods": methods}


def _metadata_float(metadata: dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _interpretation_lines(snapshot: dict[str, object]) -> list[str]:
    metrics = snapshot.get("summary_metrics", {})
    methods = metrics.get("methods", {}) if isinstance(metrics, dict) else {}
    lines: list[str] = []
    for method in ("Empirical PCA", "DRO-PCA"):
        values = methods.get(method, {}) if isinstance(methods, dict) else {}
        if not isinstance(values, dict):
            continue
        early = values.get("early_alert_rate")
        late = values.get("late_alert_rate")
        if isinstance(early, (int, float)) and isinstance(late, (int, float)):
            lines.append(
                f"* **{method}:** alert rate changes from {100.0 * float(early):.1f}% "
                f"in the first life interval to {100.0 * float(late):.1f}% in the final interval."
            )
    metadata = snapshot.get("metadata", {})
    if isinstance(metadata, dict):
        distance = _metadata_float({str(k): str(v) for k, v in metadata.items()}, "projector_distance_to_empirical")
        source = str(metadata.get("dro_selected_candidate_source", "not recorded"))
        gamma = metadata.get("dro_selected_gamma", "not recorded")
        if distance is not None and distance <= 1e-8:
            lines.append(
                "* The fitted DRO-PCA and empirical-PCA projectors are numerically equivalent "
                f"(Frobenius distance {distance:.3g}); the curves may overlap."
            )
        elif distance is not None:
            lines.append(
                f"* The fitted projectors differ by Frobenius distance {distance:.3g}."
            )
        lines.append(
            f"* DRO candidate selection: source ``{source}``, gamma ``{gamma}``."
        )
    lines.append(
        "* This snapshot validates the monitoring workflow on this fixed protocol; it does not "
        "establish universal superiority of one PCA estimator."
    )
    return lines


def _git_state(root: Path) -> tuple[str, bool]:
    """Return the current commit and whether tracked or untracked files differ."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return commit, bool(status.strip())


def _validate_source_file(path: Path) -> None:
    if path.is_symlink():
        raise SnapshotError(f"refusing to publish symbolic link: {path}")
    if not path.is_file():
        raise SnapshotError(f"required output is missing: {path}")
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise SnapshotError(f"unsupported snapshot file type: {path.name}")
    limit = MAX_IMAGE_BYTES if suffix in {".png", ".svg"} else MAX_TABLE_BYTES
    if path.stat().st_size > limit:
        raise SnapshotError(
            f"{path.name} is {path.stat().st_size} bytes, above the snapshot limit {limit}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "snapshots": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid snapshot manifest: {path}") from exc
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(payload.get("snapshots"), list):
        raise SnapshotError(f"unsupported snapshot manifest schema in {path}")
    return payload


def _page_text(profile: SnapshotProfile, snapshot: dict[str, object]) -> str:
    metadata = snapshot["metadata"]
    assert isinstance(metadata, dict)
    lines = [
        profile.title,
        "=" * len(profile.title),
        "",
        ".. note::",
        "",
        "   **Reviewed reference snapshot.** The dataset was processed locally. Read the Docs",
        "   renders committed aggregate outputs and does not download or execute this benchmark.",
        "",
        profile.description,
        "",
        "Protocol",
        "--------",
        "",
        profile.protocol + ".",
        "",
        "Interpretation",
        "--------------",
        "",
        *_interpretation_lines(snapshot),
        "",
    ]
    for filename, caption, alt in profile.figures:
        lines.extend(
            [
                f".. figure:: ../_static/external_results/{profile.slug}/{filename}",
                "   :width: 92%",
                f"   :alt: {alt}",
                "",
                f"   {caption}.",
                "",
            ]
        )
    lines.extend(["Aggregate outputs", "-----------------", ""])
    for filename in profile.tables:
        lines.append(
            f"* :download:`{filename} <../_static/external_results/{profile.slug}/{filename}>`"
        )
    lines.extend(
        [
            "",
            "Provenance",
            "----------",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Field",
            "     - Value",
        ]
    )
    public_fields = {
        "Generated (UTC)": snapshot["generated_at"],
        "Git commit": snapshot["git_commit"],
        "Command": snapshot["command"],
        "Archive SHA-256": metadata.get("archive_sha256", "not recorded"),
        "Dataset citation": metadata.get("dataset_citation", "see dataset guide"),
        "Dataset homepage": metadata.get("dataset_homepage", "see dataset guide"),
        "Requested false-alarm rate": metadata.get("false_alarm_rate", "not recorded"),
        "DRO candidate source": metadata.get("dro_selected_candidate_source", "not recorded"),
        "DRO selected gamma": metadata.get("dro_selected_gamma", "not recorded"),
        "Projector distance to empirical": metadata.get(
            "projector_distance_to_empirical", "not recorded"
        ),
    }
    literal_fields = {
        "Git commit",
        "Command",
        "Archive SHA-256",
        "DRO candidate source",
        "DRO selected gamma",
        "Projector distance to empirical",
    }
    for key, value in public_fields.items():
        safe_value = str(value).replace("\n", " ")
        lines.append(f"   * - {key}")
        lines.append(f"     - ``{safe_value}``" if key in literal_fields else f"     - {safe_value}")
    lines.extend(
        [
            "",
            "The full raw dataset, cache, row-level scores, and local filesystem paths are not",
            "included in this repository or its release artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_includes(manifest: dict[str, object]) -> tuple[str, str]:
    snapshots = manifest["snapshots"]
    assert isinstance(snapshots, list)
    cards = [
        ".. This file is generated by scripts/publish_external_snapshot.py.",
        "",
    ]
    toctree = [
        ".. This file is generated by scripts/publish_external_snapshot.py.",
    ]
    if not snapshots:
        cards.extend(
            [
                "No reviewed public-dataset snapshots have been committed yet. Run the external",
                "protocols locally, review the outputs, and publish them with",
                "``scripts/publish_external_snapshot.py``.",
                "",
            ]
        )
    else:
        cards.extend([".. raw:: html", "", "   <div class=\"gallery-grid\">"])
        toctree.extend(["", ".. toctree::", "   :maxdepth: 1", "   :hidden:", ""])
        for item in sorted(snapshots, key=lambda value: str(value["slug"])):
            slug = str(item["slug"])
            title = str(item["title"])
            preview = str(item["preview"])
            description = str(item["description"])
            cards.extend(
                [
                    f"     <a class=\"gallery-card\" href=\"external_results/{slug}.html\">",
                    f"       <img src=\"_static/external_results/{slug}/{preview}\" alt=\"{title} external benchmark snapshot\">",
                    f"       <h3>{title}</h3>",
                    f"       <p><strong>Reference snapshot.</strong> {description}</p>",
                    "     </a>",
                ]
            )
            toctree.append(f"   ../external_results/{slug}")
        cards.extend(["   </div>", ""])
    return "\n".join(cards), "\n".join(toctree) + "\n"


def _regenerate_includes(root: Path, manifest: dict[str, object]) -> None:
    cards, toctree = _render_includes(manifest)
    _atomic_write_text(root / "docs/_generated/external_snapshot_cards.rst", cards)
    _atomic_write_text(root / "docs/_generated/external_snapshot_toctree.rst", toctree)


def publish(
    root: Path,
    profile: SnapshotProfile,
    results_dir: Path,
    *,
    command: str,
    generated_at: str | None,
    replace: bool,
    allow_dirty: bool = False,
) -> Path:
    results_dir = results_dir.expanduser().resolve()
    git_commit, git_dirty = _git_state(root)
    if git_dirty and not allow_dirty:
        raise SnapshotError(
            "refusing to publish from a dirty working tree; commit the implementation "
            "or pass --allow-dirty for a local preview that will fail registry validation"
        )
    for filename in profile.required_files:
        _validate_source_file(results_dir / filename)
    metadata = _read_metadata(results_dir / "run_metadata.csv")
    for key, expected in profile.expected_metadata:
        if metadata.get(key) != expected:
            raise SnapshotError(
                f"metadata {key!r} must be {expected!r} for {profile.slug}, got {metadata.get(key)!r}"
            )
    required_metadata = {
        "false_alarm_rate",
        "dro_selected_candidate_source",
        "dro_selected_gamma",
        "projector_distance_to_empirical",
    }
    missing_metadata = sorted(required_metadata.difference(metadata))
    if missing_metadata:
        raise SnapshotError(
            "C-MAPSS results predate the reviewed evidence contract; regenerate the protocol. "
            f"Missing metadata: {missing_metadata}"
        )
    _, summary_metrics = _read_cmapss_summary(results_dir / "summary.csv")

    destination = root / "docs/_static/external_results" / profile.slug
    if destination.exists() and not replace:
        raise SnapshotError(f"snapshot already exists: {destination}; pass --replace after review")
    temporary = destination.with_name(f".{profile.slug}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    selected = tuple(item[0] for item in profile.figures) + profile.tables
    files: dict[str, dict[str, object]] = {}
    try:
        for filename in selected:
            source = results_dir / filename
            target = temporary / filename
            shutil.copyfile(source, target)
            files[filename] = {"sha256": _digest(target), "size_bytes": target.stat().st_size}
        timestamp = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        snapshot: dict[str, object] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "slug": profile.slug,
            "title": profile.title,
            "protocol": profile.protocol,
            "description": profile.description,
            "generated_at": timestamp,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "command": command,
            "metadata": metadata,
            "summary_metrics": summary_metrics,
            "files": files,
        }
        _atomic_write_json(temporary / "snapshot.json", snapshot)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    page = root / "docs/external_results" / f"{profile.slug}.rst"
    _atomic_write_text(page, _page_text(profile, snapshot))

    manifest_path = root / "docs/_static/external_results/manifest.json"
    manifest = _load_manifest(manifest_path)
    snapshots = [item for item in manifest["snapshots"] if item.get("slug") != profile.slug]
    snapshots.append(
        {
            "slug": profile.slug,
            "title": profile.title,
            "description": profile.description,
            "preview": profile.figures[0][0],
            "generated_at": snapshot["generated_at"],
            "git_commit": snapshot["git_commit"],
        }
    )
    manifest["snapshots"] = sorted(snapshots, key=lambda item: str(item["slug"]))
    _atomic_write_json(manifest_path, manifest)
    _regenerate_includes(root, manifest)
    return destination


def remove(root: Path, slug: str, *, missing_ok: bool = False) -> bool:
    """Remove one committed snapshot and regenerate the gallery registry."""

    manifest_path = root / "docs/_static/external_results/manifest.json"
    manifest = _load_manifest(manifest_path)
    snapshots = manifest["snapshots"]
    assert isinstance(snapshots, list)
    retained = [item for item in snapshots if item.get("slug") != slug]
    destination = root / "docs/_static/external_results" / slug
    page = root / "docs/external_results" / f"{slug}.rst"
    existed = len(retained) != len(snapshots) or destination.exists() or page.exists()
    if not existed and not missing_ok:
        raise SnapshotError(f"snapshot does not exist: {slug}")
    if destination.exists():
        shutil.rmtree(destination)
    if page.exists():
        page.unlink()
    manifest["snapshots"] = retained
    _atomic_write_json(manifest_path, manifest)
    _regenerate_includes(root, manifest)
    return existed


def check(
    root: Path,
    *,
    rewrite_generated: bool = False,
    required: Iterable[str] = (),
) -> None:
    manifest_path = root / "docs/_static/external_results/manifest.json"
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []
    snapshots = manifest["snapshots"]
    assert isinstance(snapshots, list)
    seen: set[str] = set()
    required_slugs = tuple(dict.fromkeys(required))
    unknown_required = sorted(set(required_slugs).difference(PROFILES))
    if unknown_required:
        errors.append(f"unknown required snapshot profiles: {unknown_required}")
    for item in snapshots:
        slug = str(item.get("slug", ""))
        if not slug or slug in seen:
            errors.append(f"invalid or duplicate snapshot slug: {slug!r}")
            continue
        seen.add(slug)
        if slug not in PROFILES:
            errors.append(f"snapshot slug is not an approved public profile: {slug!r}")
        directory = root / "docs/_static/external_results" / slug
        snapshot_path = directory / "snapshot.json"
        page = root / "docs/external_results" / f"{slug}.rst"
        if not snapshot_path.is_file():
            errors.append(f"missing {snapshot_path}")
            continue
        if not page.is_file():
            errors.append(f"missing {page}")
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {snapshot_path}: {exc}")
            continue
        if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            errors.append(
                f"snapshot {slug} uses schema {snapshot.get('schema_version')!r}; "
                "regenerate it with the current publisher"
            )
        if snapshot.get("slug") != slug:
            errors.append(f"snapshot slug mismatch for {slug}: {snapshot.get('slug')!r}")
        if snapshot.get("git_dirty") is True:
            errors.append(f"snapshot was generated from a dirty working tree: {slug}")
        profile = PROFILES.get(slug)
        metadata = snapshot.get("metadata", {})
        if profile is not None and isinstance(metadata, dict):
            for key, expected in profile.expected_metadata:
                if metadata.get(key) != expected:
                    errors.append(
                        f"snapshot metadata {key!r} must be {expected!r} for {slug}"
                    )
            required_metadata = {
                "false_alarm_rate",
                "dro_selected_candidate_source",
                "dro_selected_gamma",
                "projector_distance_to_empirical",
            }
            missing_metadata = sorted(required_metadata.difference(metadata))
            if missing_metadata:
                errors.append(f"snapshot {slug} is missing metadata: {missing_metadata}")
        summary_path = directory / "summary.csv"
        try:
            _, expected_metrics = _read_cmapss_summary(summary_path)
        except SnapshotError as exc:
            errors.append(str(exc))
        else:
            if snapshot.get("summary_metrics") != expected_metrics:
                errors.append(f"summary metrics are stale for {slug}; republish the snapshot")
        for filename, details in snapshot.get("files", {}).items():
            path = directory / filename
            try:
                _validate_source_file(path)
            except SnapshotError as exc:
                errors.append(str(exc))
                continue
            if _digest(path) != details.get("sha256"):
                errors.append(f"digest mismatch for {path}")
        forbidden = {"window_scores.csv", "raw.csv", "data.csv"}
        present = {path.name for path in directory.iterdir() if path.is_file()}
        if present & forbidden:
            errors.append(f"row-level or raw output found in {directory}: {sorted(present & forbidden)}")
        if profile is not None and isinstance(snapshot, dict):
            expected_page = _page_text(profile, snapshot)
            actual_page = page.read_text(encoding="utf-8") if page.is_file() else None
            if actual_page != expected_page:
                errors.append(f"stale generated snapshot page {page}; republish the snapshot")
    missing_required = sorted(set(required_slugs).difference(seen))
    if missing_required:
        errors.append(f"required snapshots are missing: {missing_required}")
    cards, toctree = _render_includes(manifest)
    generated = {
        root / "docs/_generated/external_snapshot_cards.rst": cards,
        root / "docs/_generated/external_snapshot_toctree.rst": toctree,
    }
    if rewrite_generated:
        for path, text in generated.items():
            _atomic_write_text(path, text)
    else:
        for path, expected in generated.items():
            actual = path.read_text(encoding="utf-8") if path.is_file() else None
            if actual != expected:
                errors.append(
                    f"stale generated include {path}; publish again or run check --rewrite-generated"
                )
    if errors:
        raise SnapshotError("snapshot validation failed:\n- " + "\n- ".join(errors))


def _root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_root_from_script(), help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list", help="list snapshot profiles")
    check_parser = subparsers.add_parser("check", help="validate committed external snapshots")
    check_parser.add_argument(
        "--rewrite-generated",
        action="store_true",
        help="refresh generated gallery cards and toctree from the manifest",
    )
    check_parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=tuple(PROFILES),
        help="require an approved snapshot slug; repeat for multiple profiles",
    )

    publish_parser = subparsers.add_parser("publish", help="publish one reviewed result directory")
    publish_parser.add_argument("dataset", choices=tuple(PROFILES))
    publish_parser.add_argument("--results", type=Path)
    publish_parser.add_argument("--command", required=True, help="exact reproduction command")
    publish_parser.add_argument("--generated-at", help="ISO-8601 timestamp; defaults to current UTC")
    publish_parser.add_argument("--replace", action="store_true")
    publish_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a local preview from uncommitted code; registry check will reject it",
    )

    remove_parser = subparsers.add_parser("remove", help="remove one published snapshot")
    remove_parser.add_argument("slug")
    remove_parser.add_argument("--missing-ok", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    try:
        if args.action == "list":
            for profile in PROFILES.values():
                print(f"{profile.slug}\t{profile.title}\t{profile.default_results}")
            return 0
        if args.action == "check":
            check(
                root,
                rewrite_generated=args.rewrite_generated,
                required=args.require,
            )
            print("external snapshot registry: OK")
            return 0
        if args.action == "remove":
            existed = remove(root, args.slug, missing_ok=args.missing_ok)
            print(f"removed,{args.slug}" if existed else f"absent,{args.slug}")
            return 0
        profile = PROFILES[args.dataset]
        results = args.results or root / profile.default_results
        destination = publish(
            root,
            profile,
            results,
            command=args.command,
            generated_at=args.generated_at,
            replace=args.replace,
            allow_dirty=args.allow_dirty,
        )
        print(f"published,{destination}")
        return 0
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
