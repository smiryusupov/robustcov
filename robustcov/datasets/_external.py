# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Safe cache and download helpers for optional external datasets.

Nothing in this module downloads during import.  Callers must explicitly request
``download=True`` (or use the command-line fetch command).  Raw archives and
extracted files live under a user cache directory, never inside the package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import new as new_hash
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile


class DatasetNotFoundError(FileNotFoundError):
    """Raised when an external dataset is not cached and download is disabled."""


class DatasetDownloadError(RuntimeError):
    """Raised when every configured upstream source fails."""


class DatasetIntegrityError(RuntimeError):
    """Raised when an archive or extracted dataset fails validation."""


@dataclass(frozen=True)
class ArchiveSource:
    """One downloadable archive and its optional published checksum."""

    url: str
    checksum: str | None = None
    algorithm: str = "sha256"
    label: str = "upstream"


@dataclass(frozen=True)
class ExternalDatasetInfo:
    """Human- and machine-readable metadata for an external dataset."""

    name: str
    slug: str
    homepage: str
    citation: str
    license_name: str
    license_url: str | None
    terms_note: str
    sources: tuple[ArchiveSource, ...]


@dataclass(frozen=True)
class PreparedExternalDataset:
    """Resolved cache paths and archive fingerprint for a dataset."""

    info: ExternalDatasetInfo
    cache_dir: Path
    archive_path: Path
    extracted_dir: Path
    archive_sha256: str
    source_url: str | None


def get_data_home(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the robustcov external-data cache directory.

    Resolution order is an explicit argument, ``ROBUSTCOV_DATA_DIR``,
    ``XDG_CACHE_HOME/robustcov``, and finally ``~/.cache/robustcov``.
    """

    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    configured = os.environ.get("ROBUSTCOV_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / "robustcov").resolve()
    return (Path.home() / ".cache" / "robustcov").resolve()


def file_digest(path: str | os.PathLike[str], algorithm: str = "sha256") -> str:
    """Return a streaming cryptographic digest for *path*."""

    digest = new_hash(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(path: Path, expected: str, algorithm: str) -> None:
    actual = file_digest(path, algorithm)
    if actual.lower() != expected.lower():
        raise DatasetIntegrityError(
            f"checksum mismatch for {path.name}: expected {algorithm}:{expected}, "
            f"got {algorithm}:{actual}"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".partial", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _download_one(source: ArchiveSource, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        source.url,
        headers={
            "User-Agent": "robustcov external-data loader (+https://github.com/smiryusupov/robustcov)",
            "Accept": "application/zip, application/octet-stream, */*",
        },
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise DatasetDownloadError(f"downloaded an empty archive from {source.url}")
        if source.checksum:
            _validate_digest(temporary, source.checksum, source.algorithm)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_local_archive(source_path: Path, destination: Path) -> None:
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"external archive does not exist: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256.json")


def _record_or_validate_local_fingerprint(path: Path, source_url: str | None) -> str:
    digest = file_digest(path, "sha256")
    sidecar = _archive_sidecar(path)
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetIntegrityError(f"invalid archive fingerprint file: {sidecar}") from exc
        expected = str(payload.get("sha256", ""))
        if expected and expected != digest:
            raise DatasetIntegrityError(
                f"cached archive {path} changed since its first verified use; "
                "remove the dataset cache and fetch it again"
            )
    else:
        _write_json_atomic(
            sidecar,
            {
                "archive": path.name,
                "sha256": digest,
                "source_url": source_url,
                "note": "Local cache fingerprint. Upstream did not provide a SHA-256 value.",
            },
        )
    return digest


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_extract_zip(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    max_members: int = 10_000,
    max_uncompressed_bytes: int = 2_000_000_000,
) -> tuple[Path, ...]:
    """Extract a ZIP archive after path, symlink, and size validation."""

    archive = Path(archive_path)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()

    try:
        zf = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetIntegrityError(f"invalid ZIP archive: {archive}") from exc

    with zf:
        members = zf.infolist()
        if len(members) > max_members:
            raise DatasetIntegrityError(
                f"archive has {len(members)} members, exceeding the safety limit {max_members}"
            )
        total = sum(info.file_size for info in members)
        if total > max_uncompressed_bytes:
            raise DatasetIntegrityError(
                f"archive expands to {total} bytes, exceeding the safety limit "
                f"{max_uncompressed_bytes}"
            )

        extracted: list[Path] = []
        for info in members:
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise DatasetIntegrityError(f"unsafe archive member path: {info.filename!r}")
            if _is_zip_symlink(info):
                raise DatasetIntegrityError(f"symbolic links are not allowed in archives: {name!r}")
            output = (target / Path(*pure.parts)).resolve()
            try:
                output.relative_to(target_root)
            except ValueError as exc:
                raise DatasetIntegrityError(f"archive member escapes target directory: {name!r}") from exc
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            extracted.append(output)
    return tuple(extracted)


def _extract_if_needed(archive: Path, extracted_dir: Path, digest: str) -> None:
    marker = extracted_dir / ".robustcov-extracted.json"
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("archive_sha256") == digest:
            return

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    files = safe_extract_zip(archive, extracted_dir)
    _write_json_atomic(
        marker,
        {
            "archive": archive.name,
            "archive_sha256": digest,
            "members": [str(path.relative_to(extracted_dir)) for path in files],
        },
    )


def find_file(root: Path, filename: str) -> Path | None:
    """Find one file recursively by exact basename."""

    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def extract_nested_archives_until(
    root: Path,
    required_filenames: Iterable[str],
    *,
    max_depth: int = 2,
) -> None:
    """Safely unpack nested ZIP files until required basenames are visible."""

    required = tuple(required_filenames)
    if all(find_file(root, name) is not None for name in required):
        return
    seen: set[Path] = set()
    for depth in range(max_depth):
        archives = sorted(path for path in root.rglob("*.zip") if path.is_file() and path not in seen)
        if not archives:
            break
        for index, archive in enumerate(archives):
            seen.add(archive)
            target = root / f"_nested_{depth}_{index}_{archive.stem}"
            safe_extract_zip(archive, target)
        if all(find_file(root, name) is not None for name in required):
            return


def prepare_external_dataset(
    info: ExternalDatasetInfo,
    *,
    archive_filename: str,
    cache_dir: str | os.PathLike[str] | None = None,
    download: bool = False,
    archive_path: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
) -> PreparedExternalDataset:
    """Resolve, optionally download, fingerprint, and safely extract a dataset."""

    data_home = get_data_home(cache_dir)
    root = data_home / info.slug
    raw_dir = root / "raw"
    extracted_dir = root / "extracted"
    cached_archive = raw_dir / archive_filename
    source_url: str | None = None

    if archive_path is not None:
        _copy_local_archive(Path(archive_path), cached_archive)
        _archive_sidecar(cached_archive).unlink(missing_ok=True)
        source_url = f"file://{Path(archive_path).expanduser().resolve()}"
    elif not cached_archive.exists():
        if not download:
            raise DatasetNotFoundError(
                f"{info.name} is not cached under {root}. Set download=True, run "
                f"'python -m robustcov.datasets fetch {info.slug}', or pass archive_path."
            )
        errors: list[str] = []
        for source in info.sources:
            try:
                _download_one(source, cached_archive, timeout)
            except (OSError, URLError, DatasetDownloadError, DatasetIntegrityError) as exc:
                errors.append(f"{source.label}: {exc}")
                cached_archive.unlink(missing_ok=True)
                continue
            source_url = source.url
            break
        else:
            details = "\n".join(f"- {error}" for error in errors)
            raise DatasetDownloadError(
                f"could not download {info.name} from any configured source:\n{details}"
            )

    # Published checksums are validated on every use.  Otherwise the first
    # successful cache use records a local SHA-256 pin and later calls verify it.
    matching_source = next(
        (source for source in info.sources if source_url == source.url and source.checksum),
        None,
    )
    if matching_source is not None:
        _validate_digest(cached_archive, matching_source.checksum or "", matching_source.algorithm)
    digest = _record_or_validate_local_fingerprint(cached_archive, source_url)
    _extract_if_needed(cached_archive, extracted_dir, digest)
    _write_json_atomic(
        root / "dataset.json",
        {
            "dataset": asdict(info),
            "archive": cached_archive.name,
            "archive_sha256": digest,
            "source_url": source_url,
        },
    )
    return PreparedExternalDataset(
        info=info,
        cache_dir=root,
        archive_path=cached_archive,
        extracted_dir=extracted_dir,
        archive_sha256=digest,
        source_url=source_url,
    )


__all__ = [
    "ArchiveSource",
    "DatasetDownloadError",
    "DatasetIntegrityError",
    "DatasetNotFoundError",
    "ExternalDatasetInfo",
    "PreparedExternalDataset",
    "extract_nested_archives_until",
    "file_digest",
    "find_file",
    "get_data_home",
    "prepare_external_dataset",
    "safe_extract_zip",
]
