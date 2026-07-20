#!/usr/bin/env python3
"""Validate robustcov source metadata and optional distribution artifacts."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, asdict
from email.parser import Parser
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tomllib
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MINIMUMS = {
    "3.12": {"numpy": "2.0.0", "scipy": "1.13.0", "scikit-learn": "1.5.0"},
    "3.13": {"numpy": "2.1.0", "scipy": "1.14.1", "scikit-learn": "1.6.0"},
    "3.14": {"numpy": "2.3.4", "scipy": "1.16.1", "scikit-learn": "1.8.0"},
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _check(checks: list[Check], name: str, condition: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=bool(condition), detail=detail))


def _runtime_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError(f"could not find a literal __version__ in {path}")


def _citation_version(path: Path) -> str:
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"could not find version in {path}")
    return match.group(1)


def _literal_string_list(tree: ast.Module, name: str) -> list[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{name} must be a literal list of strings")
        return value
    raise ValueError(f"could not find a literal {name} assignment")


def _module_bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _metadata(text: str):
    return Parser().parsestr(text)


def _safe_archive_names(names: Iterable[str]) -> bool:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            return False
    return True


def _forbidden_archive_names(names: Iterable[str]) -> list[str]:
    """Return build caches, downloaded data, or partial downloads in an artifact."""

    forbidden: list[str] = []
    for name in names:
        path = PurePosixPath(name)
        parts = set(path.parts)
        if (
            "__pycache__" in parts
            or ".pytest_cache" in parts
            or ".external-data" in parts
            or ("docs" in parts and "_build" in parts)
            or ("examples_external" in parts and "data" in parts)
            or ("results" in parts and "external" in parts)
            or path.suffix in {".pyc", ".pyo", ".partial", ".download"}
        ):
            forbidden.append(name)
    return forbidden


def _check_core_metadata(checks: list[Check], metadata, *, label: str, project: dict) -> None:
    _check(checks, f"{label}: name", metadata.get("Name") == project["name"], str(metadata.get("Name")))
    _check(
        checks,
        f"{label}: version",
        metadata.get("Version") == project["version"],
        str(metadata.get("Version")),
    )
    _check(
        checks,
        f"{label}: requires-python",
        metadata.get("Requires-Python") == project["requires-python"],
        str(metadata.get("Requires-Python")),
    )
    _check(
        checks,
        f"{label}: SPDX license",
        metadata.get("License-Expression") == "Apache-2.0",
        str(metadata.get("License-Expression")),
    )
    license_files = set(metadata.get_all("License-File", []))
    _check(
        checks,
        f"{label}: license file metadata",
        {"LICENSE", "NOTICE"}.issubset(license_files),
        repr(sorted(license_files)),
    )


def source_checks(root: Path) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    with (root / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    project = config["project"]

    required_files = [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "RELEASE.md",
        "CITATION.cff",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        "CMakeLists.txt",
        "requirements/minimum.txt",
        "docs/methods_and_references.rst",
        "docs/project_contributions.rst",
        "docs/references.bib",
        "robustcov/provenance.py",
        "robustcov/online_subspace.py",
        "docs/online_subspace_tracking.rst",
        "tests/test_online_subspace_tracking.py",
        "robustcov/datasets/__init__.py",
        "robustcov/datasets/gas_sensor_drift.py",
        "robustcov/datasets/cmapss.py",
        "docs/external_data.rst",
        "docs/external_snapshot_policy.rst",
        "docs/_static/external_results/manifest.json",
        "scripts/publish_external_snapshot.py",
        "examples_external/gas_sensor_drift_dro_pca.py",
        "examples_external/cmapss_dro_pca_monitoring.py",
        ".github/workflows/external-data.yml",
        ".gitignore",
    ]
    missing = [name for name in required_files if not (root / name).is_file()]
    _check(checks, "source: required files", not missing, f"missing={missing}")

    runtime_version = _runtime_version(root / "robustcov" / "__init__.py")
    citation_version = _citation_version(root / "CITATION.cff")
    _check(
        checks,
        "source: version consistency",
        project["version"] == runtime_version == citation_version,
        f"pyproject={project['version']}, runtime={runtime_version}, citation={citation_version}",
    )
    _check(checks, "source: supported Python", project.get("requires-python") == ">=3.12", repr(project.get("requires-python")))
    _check(checks, "source: SPDX license", project.get("license") == "Apache-2.0", repr(project.get("license")))
    _check(
        checks,
        "source: declared license files",
        set(project.get("license-files", [])) == {"LICENSE", "NOTICE"},
        repr(project.get("license-files")),
    )
    legacy_license_classifiers = [
        item for item in project.get("classifiers", []) if item.startswith("License ::")
    ]
    _check(
        checks,
        "source: no legacy license classifier",
        not legacy_license_classifiers,
        repr(legacy_license_classifiers),
    )
    urls = project.get("urls", {})
    insecure_urls = {name: url for name, url in urls.items() if not str(url).startswith("https://")}
    _check(checks, "source: HTTPS project URLs", not insecure_urls, repr(insecure_urls))

    expected_runtime = {
        "numpy>=2.0.0; python_version < '3.13'",
        "numpy>=2.1.0; python_version >= '3.13' and python_version < '3.14'",
        "numpy>=2.3.4; python_version >= '3.14'",
        "scipy>=1.13.0; python_version < '3.13'",
        "scipy>=1.14.1; python_version >= '3.13' and python_version < '3.14'",
        "scipy>=1.16.1; python_version >= '3.14'",
    }
    runtime_dependencies = set(project.get("dependencies", []))
    _check(
        checks,
        "source: version-specific runtime lower bounds",
        runtime_dependencies == expected_runtime,
        repr(sorted(runtime_dependencies)),
    )

    expected_build_numpy = {
        "numpy>=2.0.0; python_version < '3.13'",
        "numpy>=2.1.0; python_version >= '3.13' and python_version < '3.14'",
        "numpy>=2.3.4; python_version >= '3.14'",
    }
    build_requirements = set(config.get("build-system", {}).get("requires", []))
    _check(
        checks,
        "source: version-specific build NumPy lower bounds",
        expected_build_numpy.issubset(build_requirements),
        repr(sorted(build_requirements)),
    )

    expected_minimum_lines = {
        'numpy==2.0.0; python_version < "3.13"',
        'scipy==1.13.0; python_version < "3.13"',
        'scikit-learn==1.5.0; python_version < "3.13"',
        'numpy==2.1.0; python_version >= "3.13" and python_version < "3.14"',
        'scipy==1.14.1; python_version >= "3.13" and python_version < "3.14"',
        'scikit-learn==1.6.0; python_version >= "3.13" and python_version < "3.14"',
        'numpy==2.3.4; python_version >= "3.14"',
        'scipy==1.16.1; python_version >= "3.14"',
        'scikit-learn==1.8.0; python_version >= "3.14"',
    }
    minimum_lines = {
        line.strip()
        for line in (root / "requirements" / "minimum.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    _check(
        checks,
        "source: exact minimum dependency pins",
        minimum_lines == expected_minimum_lines,
        f"declared={sorted(minimum_lines)}",
    )

    extras = project.get("optional-dependencies", {})
    required_extras = {"plot", "sklearn", "test", "dev", "bench", "docs", "examples"}
    _check(
        checks,
        "source: expected extras",
        required_extras.issubset(extras),
        f"available={sorted(extras)}",
    )

    changelog = (root / "CHANGELOG.md").read_text()
    _check(checks, "source: unreleased changelog", "## Unreleased" in changelog, "CHANGELOG.md")
    deprecation_policy = (root / "docs" / "api_stability.rst").read_text()
    _check(
        checks,
        "source: deprecation policy",
        "Deprecation policy" in deprecation_policy and "DeprecationWarning" in deprecation_policy,
        "docs/api_stability.rst",
    )

    init_path = root / "robustcov" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exported = _literal_string_list(init_tree, "__all__")
    bound_names = _module_bound_names(init_tree)
    missing_exports = [name for name in exported if name not in bound_names]
    _check(checks, "source: unique public exports", len(exported) == len(set(exported)), f"count={len(exported)}")
    _check(checks, "source: statically resolvable public exports", not missing_exports, f"missing={missing_exports}")

    return checks, config


def wheel_checks(path: Path, project: dict) -> list[Check]:
    checks: list[Check] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _check(checks, f"wheel {path.name}: safe paths", _safe_archive_names(names), "archive paths")
        forbidden = _forbidden_archive_names(names)
        _check(
            checks,
            f"wheel {path.name}: no caches or external data",
            not forbidden,
            repr(forbidden[:20]),
        )
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        _check(checks, f"wheel {path.name}: one METADATA", len(metadata_names) == 1, repr(metadata_names))
        if len(metadata_names) != 1:
            return checks
        metadata = _metadata(archive.read(metadata_names[0]).decode("utf-8"))
        _check_core_metadata(checks, metadata, label=f"wheel {path.name}", project=project)
        dist_info = metadata_names[0].rsplit("/", 1)[0]
        required_members = {
            "robustcov/__init__.py",
            "robustcov/datasets/__init__.py",
            "robustcov/datasets/_external.py",
            "robustcov/datasets/gas_sensor_drift.py",
            "robustcov/datasets/cmapss.py",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/NOTICE",
        }
        missing = sorted(required_members.difference(names))
        _check(checks, f"wheel {path.name}: required members", not missing, f"missing={missing}")
    return checks


def sdist_checks(path: Path, project: dict) -> list[Check]:
    checks: list[Check] = []
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _check(checks, f"sdist {path.name}: safe paths", _safe_archive_names(names), "archive paths")
        forbidden = _forbidden_archive_names(names)
        _check(
            checks,
            f"sdist {path.name}: no caches or external data",
            not forbidden,
            repr(forbidden[:20]),
        )
        top_levels = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        _check(checks, f"sdist {path.name}: one top-level directory", len(top_levels) == 1, repr(sorted(top_levels)))
        if len(top_levels) != 1:
            return checks
        top = next(iter(top_levels))
        required = {
            f"{top}/PKG-INFO",
            f"{top}/pyproject.toml",
            f"{top}/LICENSE",
            f"{top}/NOTICE",
            f"{top}/CMakeLists.txt",
            f"{top}/src/robustcov_cpp.cpp",
            f"{top}/robustcov/__init__.py",
            f"{top}/robustcov/provenance.py",
            f"{top}/docs/methods_and_references.rst",
            f"{top}/docs/project_contributions.rst",
            f"{top}/docs/references.bib",
            f"{top}/scripts/package_smoke_test.py",
            f"{top}/scripts/release_check.py",
            f"{top}/requirements/minimum.txt",
            f"{top}/robustcov/datasets/__init__.py",
            f"{top}/robustcov/datasets/_external.py",
            f"{top}/robustcov/datasets/gas_sensor_drift.py",
            f"{top}/robustcov/datasets/cmapss.py",
            f"{top}/examples_external/gas_sensor_drift_dro_pca.py",
            f"{top}/examples_external/cmapss_dro_pca_monitoring.py",
            f"{top}/docs/external_data.rst",
            f"{top}/docs/external_snapshot_policy.rst",
            f"{top}/docs/_static/external_results/manifest.json",
            f"{top}/scripts/publish_external_snapshot.py",
            f"{top}/docs/external_data/gas_sensor_drift.rst",
            f"{top}/docs/external_data/cmapss.rst",
            f"{top}/.github/workflows/external-data.yml",
        }
        missing = sorted(required.difference(names))
        _check(checks, f"sdist {path.name}: required members", not missing, f"missing={missing}")
        member = archive.extractfile(f"{top}/PKG-INFO")
        if member is None:
            return checks
        metadata = _metadata(member.read().decode("utf-8"))
        _check_core_metadata(checks, metadata, label=f"sdist {path.name}", project=project)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", type=Path, help="wheel or .tar.gz artifacts to inspect")
    parser.add_argument("--root", type=Path, default=ROOT, help="project source root")
    parser.add_argument("--json-output", type=Path, help="write machine-readable results")
    args = parser.parse_args()

    root = args.root.resolve()
    checks, config = source_checks(root)
    project = config["project"]
    for artifact in args.artifacts:
        artifact = artifact.resolve()
        if not artifact.is_file():
            checks.append(Check(f"artifact {artifact}", False, "file does not exist"))
        elif artifact.suffix == ".whl":
            checks.extend(wheel_checks(artifact, project))
        elif artifact.name.endswith(".tar.gz"):
            checks.extend(sdist_checks(artifact, project))
        else:
            checks.append(Check(f"artifact {artifact.name}", False, "unsupported artifact type"))

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}: {check.name}: {check.detail}")

    payload = {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "minimum_dependencies": MINIMUMS,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
