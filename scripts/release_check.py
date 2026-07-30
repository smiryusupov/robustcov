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
import subprocess
import sys
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


_FORBIDDEN_ARCHIVE_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "__pycache__",
    "_skbuild",
    "CMakeFiles",
    "build",
    "dist",
    "wheelhouse",
    "release-metadata",
    "native-free-wheel",
    "native-wheel",
    ".external-data",
}
_FORBIDDEN_ARCHIVE_FILENAMES = {
    ".coverage",
    ".DS_Store",
    "Thumbs.db",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "install_manifest.txt",
    ".ninja_deps",
    ".ninja_log",
}
_FORBIDDEN_ARCHIVE_SUFFIXES = {".pyc", ".pyo", ".partial", ".download"}
_FORBIDDEN_SDIST_SUFFIXES = {
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".whl",
    ".zip",
}


def _forbidden_archive_names(names: Iterable[str], *, artifact_kind: str) -> list[str]:
    """Return local state, build output, or nested artifacts in a distribution."""

    forbidden: list[str] = []
    for name in names:
        path = PurePosixPath(name)
        parts = set(path.parts)
        local_environment = any(part.startswith(".venv-") for part in path.parts)
        coverage_file = path.name.startswith(".coverage.")
        generated_docs = "docs" in parts and "_build" in parts
        downloaded_data = "examples_external" in parts and "data" in parts
        external_results = "results" in parts and "external" in parts
        forbidden_sdist_binary = artifact_kind == "sdist" and (
            path.suffix.lower() in _FORBIDDEN_SDIST_SUFFIXES
            or path.name.endswith(".tar.gz")
        )
        if (
            parts.intersection(_FORBIDDEN_ARCHIVE_PARTS)
            or local_environment
            or path.name in _FORBIDDEN_ARCHIVE_FILENAMES
            or coverage_file
            or generated_docs
            or downloaded_data
            or external_results
            or path.suffix.lower() in _FORBIDDEN_ARCHIVE_SUFFIXES
            or forbidden_sdist_binary
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
        "docs/faq.rst",
        "docs/_static/brand/robustcov-mark.png",
        "docs/_static/brand/robustcov-lockup.png",
        "docs/_static/brand/robustcov-favicon.png",
        "docs/references.bib",
        "robustcov/provenance.py",
        "robustcov/_public_api.json",
        "robustcov/decomposition.py",
        "docs/principal_component_pursuit.rst",
        "tests/test_principal_component_pursuit.py",
        "robustcov/online_subspace.py",
        "robustcov/experimental/spectral_filter_covariance.py",
        "docs/adversarial_covariance_filtering.rst",
        "tests/test_spectral_filter_covariance.py",
        "docs/online_subspace_tracking.rst",
        "tests/test_online_subspace_tracking.py",
        "robustcov/datasets/__init__.py",
        "robustcov/datasets/gas_sensor_drift.py",
        "robustcov/datasets/cmapss.py",
        "docs/external_data.rst",
        "docs/external_snapshot_policy.rst",
        "docs/_static/external_results/manifest.json",
        "scripts/publish_external_snapshot.py",
        "scripts/check_release_version.py",
        "scripts/installed_package_smoke.py",
        "scripts/write_artifact_checksums.py",
        "scripts/generate_release_evidence.py",
        "docs/_static/release_evidence.json",
        "docs/_static/benchmarks/statistical_validation.json",
        "docs/_static/examples/robust_explanations_iris.json",
        "examples_external/gas_sensor_drift_dro_pca.py",
        "examples_external/cmapss_dro_pca_monitoring.py",
        ".github/workflows/external-data.yml",
        ".gitignore",
    ]
    missing = [name for name in required_files if not (root / name).is_file()]
    _check(checks, "source: required files", not missing, f"missing={missing}")
    _check(
        checks,
        "source: contributor policy is not public navigation",
        not (root / "docs" / "project_contributions.rst").exists()
        and "project_contributions" not in (root / "docs" / "index.rst").read_text(encoding="utf-8"),
        "docs/project_contributions.rst",
    )

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
    _check(
        checks,
        "source: explicit-sdist build backend",
        "scikit-build-core>=1.0.3" in build_requirements,
        repr(sorted(build_requirements)),
    )
    sdist_config = config.get("tool", {}).get("scikit-build", {}).get("sdist", {})
    _check(
        checks,
        "source: explicit clean sdist inclusion",
        sdist_config.get("inclusion-mode") == "explicit"
        and "pyproject.toml" in sdist_config.get("include", []),
        repr(sdist_config.get("inclusion-mode")),
    )
    required_sdist_exclusions = {
        "**/.git/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/*.so",
        "**/*.whl",
        "**/*.zip",
        "docs/_build/**",
        "build/**",
        "dist/**",
    }
    configured_sdist_exclusions = set(sdist_config.get("exclude", []))
    _check(
        checks,
        "source: sdist excludes local and build state",
        required_sdist_exclusions.issubset(configured_sdist_exclusions),
        f"missing={sorted(required_sdist_exclusions - configured_sdist_exclusions)}",
    )
    _check(
        checks,
        "source: one sdist configuration",
        not (root / "MANIFEST.in").exists(),
        "pyproject.toml is the only sdist manifest",
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
    required_extras = {"plot", "sklearn", "test", "dev", "bench", "docs", "examples", "explain"}
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
    _check(
        checks,
        "source: one API stability policy",
        not (root / "docs" / "API_STABILITY.md").exists(),
        "docs/api_stability.rst is canonical",
    )

    evidence_check = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "generate_release_evidence.py"),
            "--check",
            "--manifest",
            str(root / "docs" / "_static" / "release_evidence.json"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    _check(
        checks,
        "source: release evidence manifest",
        evidence_check.returncode == 0,
        (evidence_check.stdout + evidence_check.stderr).strip(),
    )

    init_path = root / "robustcov" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exported = _literal_string_list(init_tree, "__all__")
    bound_names = _module_bound_names(init_tree)
    missing_exports = [name for name in exported if name not in bound_names]
    accidental_public_bindings = sorted(
        name for name in bound_names if not name.startswith("_") and name not in exported
    )
    _check(checks, "source: unique public exports", len(exported) == len(set(exported)), f"count={len(exported)}")
    _check(checks, "source: statically resolvable public exports", not missing_exports, f"missing={missing_exports}")
    _check(
        checks,
        "source: no accidental package-root bindings",
        not accidental_public_bindings,
        f"unexpected={accidental_public_bindings}",
    )

    api_manifest_path = root / "robustcov" / "_public_api.json"
    try:
        api_manifest = json.loads(api_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _check(checks, "source: public API manifest", False, str(exc))
    else:
        tiers = (
            "stable_top_level",
            "provisional_top_level",
            "experimental_top_level",
        )
        tier_values = [api_manifest.get(name, []) for name in tiers]
        valid_tiers = all(
            isinstance(values, list) and all(isinstance(item, str) for item in values)
            for values in tier_values
        )
        flattened = [item for values in tier_values if isinstance(values, list) for item in values]
        _check(checks, "source: public API manifest schema", api_manifest.get("schema_version") == 1 and valid_tiers, repr(api_manifest.get("schema_version")))
        _check(checks, "source: public API manifest version", api_manifest.get("package_version") == project["version"], repr(api_manifest.get("package_version")))
        _check(checks, "source: public API tier partition", len(flattened) == len(set(flattened)) and set(flattened) == set(exported), f"manifest={len(flattened)}, exports={len(exported)}")

        experimental_path = root / "robustcov" / "experimental" / "__init__.py"
        experimental_tree = ast.parse(experimental_path.read_text(encoding="utf-8"), filename=str(experimental_path))
        experimental_exports = _literal_string_list(experimental_tree, "__all__")
        experimental_bound_names = _module_bound_names(experimental_tree)
        missing_experimental_exports = [
            name for name in experimental_exports if name not in experimental_bound_names
        ]
        accidental_experimental_bindings = sorted(
            name
            for name in experimental_bound_names
            if not name.startswith("_") and name not in experimental_exports
        )
        manifest_experimental = api_manifest.get("experimental_namespace", [])
        _check(checks, "source: experimental namespace manifest", isinstance(manifest_experimental, list) and set(manifest_experimental) == set(experimental_exports), f"manifest={manifest_experimental}, exports={experimental_exports}")
        _check(
            checks,
            "source: statically resolvable experimental exports",
            not missing_experimental_exports,
            f"missing={missing_experimental_exports}",
        )
        _check(
            checks,
            "source: no accidental experimental bindings",
            not accidental_experimental_bindings,
            f"unexpected={accidental_experimental_bindings}",
        )

    return checks, config


def wheel_checks(path: Path, project: dict) -> list[Check]:
    checks: list[Check] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _check(checks, f"wheel {path.name}: safe paths", _safe_archive_names(names), "archive paths")
        forbidden = _forbidden_archive_names(names, artifact_kind="wheel")
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
        allowed_roots = {"robustcov", dist_info}
        unexpected_roots = sorted(
            {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts
            }
            - allowed_roots
        )
        _check(
            checks,
            f"wheel {path.name}: runtime-only roots",
            not unexpected_roots,
            f"unexpected={unexpected_roots}",
        )
        required_members = {
            "robustcov/__init__.py",
            "robustcov/_public_api.json",
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
        forbidden = _forbidden_archive_names(names, artifact_kind="sdist")
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
        allowed_children = {
            ".github",
            ".readthedocs.yaml",
            "CHANGELOG.md",
            "CITATION.cff",
            "CMakeLists.txt",
            "CONTRIBUTING.md",
            "LICENSE",
            "NOTICE",
            "PKG-INFO",
            "README.md",
            "RELEASE.md",
            "benchmarks",
            "conda",
            "docs",
            "examples",
            "examples_external",
            "notebooks",
            "pyproject.toml",
            "requirements",
            "robustcov",
            "scripts",
            "src",
            "tests",
        }
        children = {
            PurePosixPath(name).parts[1]
            for name in names
            if len(PurePosixPath(name).parts) > 1
        }
        unexpected_children = sorted(children - allowed_children)
        _check(
            checks,
            f"sdist {path.name}: allowlisted top-level contents",
            not unexpected_children,
            f"unexpected={unexpected_children}",
        )
        required = {
            f"{top}/PKG-INFO",
            f"{top}/pyproject.toml",
            f"{top}/LICENSE",
            f"{top}/NOTICE",
            f"{top}/CMakeLists.txt",
            f"{top}/src/robustcov_cpp.cpp",
            f"{top}/robustcov/__init__.py",
            f"{top}/robustcov/provenance.py",
            f"{top}/robustcov/_public_api.json",
            f"{top}/docs/methods_and_references.rst",
            f"{top}/docs/faq.rst",
            f"{top}/docs/_static/brand/robustcov-mark.png",
            f"{top}/docs/_static/brand/robustcov-lockup.png",
            f"{top}/docs/_static/brand/robustcov-favicon.png",
            f"{top}/docs/references.bib",
            f"{top}/scripts/package_smoke_test.py",
            f"{top}/scripts/release_check.py",
            f"{top}/scripts/check_release_version.py",
            f"{top}/scripts/installed_package_smoke.py",
            f"{top}/scripts/write_artifact_checksums.py",
            f"{top}/scripts/generate_release_evidence.py",
            f"{top}/docs/_static/release_evidence.json",
            f"{top}/docs/_static/benchmarks/statistical_validation.json",
            f"{top}/docs/_static/examples/robust_explanations_iris.json",
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


def release_candidate_checks(root: Path) -> list[Check]:
    """Validate evidence required for a public release candidate."""

    checks: list[Check] = []
    manifest_path = root / "docs" / "_static" / "external_results" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _check(checks, "release candidate: external snapshot manifest", False, str(exc))
        return checks

    snapshots = manifest.get("snapshots", [])
    slugs = {str(item.get("slug")) for item in snapshots if isinstance(item, dict)}
    required = {"cmapss_fd002", "cmapss_fd004"}
    _check(
        checks,
        "release candidate: required C-MAPSS snapshots",
        required.issubset(slugs),
        f"present={sorted(slugs)}, required={sorted(required)}",
    )

    dirty: list[str] = []
    unknown_commits: list[str] = []
    stale_schema: list[str] = []
    incomplete_evidence: list[str] = []
    non_ancestor_commits: list[str] = []
    for slug in sorted(required.intersection(slugs)):
        snapshot_path = root / "docs" / "_static" / "external_results" / slug / "snapshot.json"
        page_path = root / "docs" / "external_results" / f"{slug}.rst"
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dirty.append(slug)
            continue
        if snapshot.get("schema_version") != 2:
            stale_schema.append(slug)
        if snapshot.get("git_dirty") is not False:
            dirty.append(slug)
        commit = snapshot.get("git_commit")
        if commit in {None, "", "unknown"}:
            unknown_commits.append(slug)
        elif (root / ".git").exists():
            try:
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", str(commit), "HEAD"],
                    cwd=root,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                non_ancestor_commits.append(slug)
        metadata = snapshot.get("metadata", {})
        summary_metrics = snapshot.get("summary_metrics", {})
        required_metadata = {
            "false_alarm_rate",
            "dro_selected_candidate_source",
            "dro_selected_gamma",
            "projector_distance_to_empirical",
        }
        methods = summary_metrics.get("methods", {}) if isinstance(summary_metrics, dict) else {}
        if (
            not isinstance(metadata, dict)
            or not required_metadata.issubset(metadata)
            or not isinstance(methods, dict)
            or not {"Empirical PCA", "DRO-PCA"}.issubset(methods)
            or not page_path.is_file()
        ):
            incomplete_evidence.append(slug)
    _check(checks, "release candidate: current snapshot schema", not stale_schema, f"invalid={stale_schema}")
    _check(checks, "release candidate: complete snapshot evidence", not incomplete_evidence, f"invalid={incomplete_evidence}")
    _check(checks, "release candidate: clean snapshot provenance", not dirty, f"invalid={dirty}")
    _check(checks, "release candidate: snapshot commits recorded", not unknown_commits, f"invalid={unknown_commits}")
    _check(checks, "release candidate: snapshot commits are ancestors", not non_ancestor_commits, f"invalid={non_ancestor_commits}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", type=Path, help="wheel or .tar.gz artifacts to inspect")
    parser.add_argument("--root", type=Path, default=ROOT, help="project source root")
    parser.add_argument("--json-output", type=Path, help="write machine-readable results")
    parser.add_argument(
        "--release-candidate",
        action="store_true",
        help="also require reviewed FD002/FD004 snapshots with clean provenance",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    checks, config = source_checks(root)
    if args.release_candidate:
        checks.extend(release_candidate_checks(root))
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
