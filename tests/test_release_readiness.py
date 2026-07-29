# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_pep639_license_metadata_and_required_files():
    project = _config()["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) == {"LICENSE", "NOTICE"}
    assert not any(item.startswith("License ::") for item in project["classifiers"])
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "NOTICE").is_file()


def test_version_specific_runtime_lower_bounds_are_declared():
    dependencies = set(_config()["project"]["dependencies"])
    assert dependencies == {
        "numpy>=2.0.0; python_version < '3.13'",
        "numpy>=2.1.0; python_version >= '3.13' and python_version < '3.14'",
        "numpy>=2.3.4; python_version >= '3.14'",
        "scipy>=1.13.0; python_version < '3.13'",
        "scipy>=1.14.1; python_version >= '3.13' and python_version < '3.14'",
        "scipy>=1.16.1; python_version >= '3.14'",
    }


def test_minimum_requirements_cover_all_supported_pythons():
    text = (ROOT / "requirements" / "minimum.txt").read_text()
    for requirement in (
        "numpy==2.0.0",
        "scipy==1.13.0",
        "scikit-learn==1.5.0",
        "numpy==2.1.0",
        "scipy==1.14.1",
        "scikit-learn==1.6.0",
        "numpy==2.3.4",
        "scipy==1.16.1",
        "scikit-learn==1.8.0",
    ):
        assert requirement in text


def test_optional_matplotlib_conftest_import():
    text = (ROOT / "tests" / "conftest.py").read_text()
    assert "except ModuleNotFoundError" in text
    assert 'matplotlib.use("Agg"' in text


def test_public_deprecation_policy_is_documented():
    text = (ROOT / "docs" / "api_stability.rst").read_text()
    assert "Deprecation policy" in text
    assert "DeprecationWarning" in text
    assert "CHANGELOG.md" in text


def test_release_check_source_mode(tmp_path):
    output = tmp_path / "release-check.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "release_check.py"),
            "--root",
            str(ROOT),
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["minimum_dependencies"]["3.14"]["scipy"] == "1.16.1"


def test_release_check_runs_without_site_packages(tmp_path):
    output = tmp_path / "release-check-no-site.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts" / "release_check.py"),
            "--root",
            str(ROOT),
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["passed"] is True
