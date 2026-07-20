#!/usr/bin/env python3
"""Check that all release-version declarations agree with an optional Git tag."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(
    r"^(?P<release>0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:(?:a|b|rc)(?:0|[1-9]\d*))?(?:\.post(?:0|[1-9]\d*))?"
    r"(?:\.dev(?:0|[1-9]\d*))?$"
)


class VersionError(RuntimeError):
    """Raised when release version declarations are inconsistent."""


def _runtime_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise VersionError(f"could not find a literal __version__ in {path}")


def _citation_version(path: Path) -> str:
    match = re.search(
        r"^version:\s*[\"']?([^\"'\s]+)",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise VersionError(f"could not find version in {path}")
    return match.group(1)


def declared_versions(root: Path) -> dict[str, str]:
    with (root / "pyproject.toml").open("rb") as handle:
        project_version = str(tomllib.load(handle)["project"]["version"])
    api_manifest = json.loads(
        (root / "robustcov/_public_api.json").read_text(encoding="utf-8")
    )
    return {
        "pyproject.toml": project_version,
        "robustcov.__version__": _runtime_version(root / "robustcov/__init__.py"),
        "CITATION.cff": _citation_version(root / "CITATION.cff"),
        "robustcov/_public_api.json": str(api_manifest.get("package_version", "")),
    }


def check_versions(
    root: Path,
    *,
    tag: str | None = None,
    expected: str | None = None,
    require_prerelease: bool = False,
) -> str:
    versions = declared_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in versions.items())
        raise VersionError(f"release versions disagree: {detail}")
    version = next(iter(unique))
    if not VERSION_RE.fullmatch(version):
        raise VersionError(f"unsupported release version syntax: {version!r}")
    if expected is not None and version != expected:
        raise VersionError(f"expected version {expected!r}, found {version!r}")
    if tag is not None:
        normalized_tag = tag.removeprefix("refs/tags/")
        if normalized_tag != f"v{version}":
            raise VersionError(
                f"release tag {normalized_tag!r} does not match package version v{version}"
            )
    if require_prerelease and not re.search(r"(?:a|b|rc)\d+", version):
        raise VersionError(f"TestPyPI rehearsal requires a pre-release version, found {version!r}")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="require a matching v<version> tag")
    parser.add_argument("--expected", help="require this exact version")
    parser.add_argument("--require-prerelease", action="store_true")
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()
    try:
        version = check_versions(
            args.root.resolve(),
            tag=args.tag,
            expected=args.expected,
            require_prerelease=args.require_prerelease,
        )
    except (OSError, ValueError, json.JSONDecodeError, VersionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.print_version:
        print(version)
    else:
        print(f"release version: OK ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
