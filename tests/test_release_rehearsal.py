from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERSION_SCRIPT = ROOT / "scripts/check_release_version.py"
CHECKSUM_SCRIPT = ROOT / "scripts/write_artifact_checksums.py"
SMOKE_SCRIPT = ROOT / "scripts/installed_package_smoke.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_version_declarations_agree_with_alpha_version() -> None:
    checker = _load_script("check_release_version_test", VERSION_SCRIPT)
    assert checker.check_versions(
        ROOT,
        expected="0.1.0a2",
        require_prerelease=True,
    ) == "0.1.0a2"
    assert checker.check_versions(ROOT, tag="v0.1.0a2") == "0.1.0a2"
    with pytest.raises(checker.VersionError, match="does not match"):
        checker.check_versions(ROOT, tag="v0.1.0a3")


def test_release_version_cli_prints_only_the_version() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERSION_SCRIPT), "--print-version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "0.1.0a2"


def test_checksum_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "b.whl"
    second = tmp_path / "a.tar.gz"
    first.write_bytes(b"wheel")
    second.write_bytes(b"sdist")
    output = tmp_path / "SHA256SUMS"
    subprocess.run(
        [
            sys.executable,
            str(CHECKSUM_SCRIPT),
            str(first),
            str(second),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["a.tar.gz", "b.whl"]
    assert all(len(line.split("  ", 1)[0]) == 64 for line in lines)


def test_release_workflow_has_separate_trusted_publish_jobs() -> None:
    workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
    assert "publish_target:" in workflow
    assert "environment:\n      name: testpypi" in workflow
    assert "repository-url: https://test.pypi.org/legacy/" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert workflow.count("id-token: write") == 2
    assert "Smoke-test TestPyPI installation" in workflow
    assert "scripts/check_release_version.py --tag" in workflow
    assert "scripts/release_check.py --release-candidate" in workflow
    assert "fetch-depth: 0" in workflow
    assert "actions/download-artifact@v6" in workflow


def test_release_helpers_are_shipped_in_source_distribution_configuration() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"scripts/**"' in pyproject
    release_checker = (ROOT / "scripts/release_check.py").read_text(encoding="utf-8")
    for name in (
        "scripts/check_release_version.py",
        "scripts/installed_package_smoke.py",
        "scripts/write_artifact_checksums.py",
    ):
        assert name in release_checker


def test_installed_smoke_helper_has_no_source_tree_import_side_effects() -> None:
    completed = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--forbid-root" in completed.stdout


def test_public_api_manifest_uses_release_candidate_version() -> None:
    payload = json.loads((ROOT / "robustcov/_public_api.json").read_text(encoding="utf-8"))
    assert payload["package_version"] == "0.1.0a2"


def test_sdist_uses_explicit_allowlist_with_effective_exclusions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert "scikit-build-core>=1.0.3" in config["build-system"]["requires"]
    sdist = config["tool"]["scikit-build"]["sdist"]
    assert sdist["inclusion-mode"] == "explicit"
    assert "pyproject.toml" in sdist["include"]
    assert "**/__pycache__/**" in sdist["exclude"]
