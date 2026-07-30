# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import tomllib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_python_package(tmp_path: Path) -> Path:
    package_root = tmp_path / "source"
    destination = package_root / "robustcov"
    shutil.copytree(PROJECT_ROOT / "robustcov", destination)
    for native_artifact in destination.glob("_robustcov_cpp*"):
        native_artifact.unlink()
    return package_root


def _run_isolated(package_root: Path, code: str) -> subprocess.CompletedProcess[str]:
    # ``-S`` prevents editable-install ``.pth`` hooks from registering import
    # finders that can shadow the copied package or leak a compiled extension
    # from the developer environment. Add the normal dependency directories
    # back explicitly without executing their ``.pth`` files.
    dependency_paths = tuple(
        dict.fromkeys(
            path
            for path in sys.path
            if path and Path(path).name in {"site-packages", "dist-packages"}
        )
    )
    bootstrap = f"""
import sys
for dependency_path in {dependency_paths!r}:
    if dependency_path not in sys.path:
        sys.path.append(dependency_path)
sys.path.insert(0, {str(package_root)!r})
"""
    return subprocess.run(
        [sys.executable, "-S", "-c", bootstrap + textwrap.dedent(code)],
        cwd=package_root.parent,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_matplotlib_is_an_optional_dependency():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    dependencies = config["project"]["dependencies"]
    plot_dependencies = config["project"]["optional-dependencies"]["plot"]
    assert not any(item.lower().startswith("matplotlib") for item in dependencies)
    assert any(item.lower().startswith("matplotlib") for item in plot_dependencies)


def test_explainers_are_optional_dependencies():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    dependencies = config["project"]["dependencies"]
    explain_dependencies = config["project"]["optional-dependencies"]["explain"]
    assert not any(item.lower().startswith(("shap", "lime")) for item in dependencies)
    assert any(item.lower().startswith("shap") for item in explain_dependencies)
    assert any(item.lower().startswith("lime") for item in explain_dependencies)


def test_import_without_matplotlib_and_clear_plotting_error(tmp_path):
    package_root = _copy_python_package(tmp_path)
    result = _run_isolated(
        package_root,
        """
        import importlib.abc
        import sys

        class BlockMatplotlib(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("matplotlib blocked for test")
                return None

        sys.meta_path.insert(0, BlockMatplotlib())
        import robustcov as rc
        assert rc.__version__
        try:
            rc.plot_covariance_heatmap([[1.0]], show=False)
        except ImportError as exc:
            message = str(exc)
            assert "robustcov[plot]" in message
            assert "matplotlib" in message
        else:
            raise AssertionError("plotting should require the optional dependency")
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_free_import_and_python_estimators(tmp_path):
    package_root = _copy_python_package(tmp_path)
    result = _run_isolated(
        package_root,
        """
        import sys

        # scikit-build editable installs register a meta-path finder that can
        # expose the compiled extension from the development build even when
        # this copied package contains no native artifact. An explicit None
        # entry is the standard import-system sentinel for an unavailable
        # module and prevents that external editable finder from leaking into
        # this native-free subprocess.
        sys.modules["robustcov._robustcov_cpp"] = None

        import numpy as np
        import robustcov as rc

        assert rc.native_available() is False
        assert rc.has_openmp() is False
        assert rc.get_num_threads() == 1
        rc.set_num_threads(1)

        try:
            rc.set_num_threads(2)
        except RuntimeError as exc:
            assert "native extension is unavailable" in str(exc)
        else:
            raise AssertionError("native-free thread control should reject n_threads > 1")

        X = np.random.default_rng(0).normal(size=(50, 4))
        fitted = rc.RegularizedCauchy(alpha=0.1, max_iter=30).fit(X)
        assert fitted.covariance_.shape == (4, 4)

        try:
            rc.FastMCD(n_init=5, random_state=0).fit(X)
        except RuntimeError as exc:
            message = str(exc)
            assert "FastMCD.fit requires the robustcov native extension" in message
            assert "supported binary wheel" in message
        else:
            raise AssertionError("FastMCD should fail clearly without native code")
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stale_native_extension_is_rejected_with_rebuild_guidance(tmp_path):
    package_root = _copy_python_package(tmp_path)
    fake_native = package_root / "robustcov" / "_robustcov_cpp.py"
    fake_native.write_text(
        "def has_openmp():\n"
        "    return True\n",
        encoding="utf-8",
    )
    result = _run_isolated(
        package_root,
        """
        import numpy as np
        import robustcov as rc

        assert rc.native_available() is False
        try:
            rc.FastMCD(n_init=5, random_state=0).fit(
                np.random.default_rng(0).normal(size=(50, 4))
            )
        except RuntimeError as exc:
            message = str(exc)
            assert "incompatible robustcov native extension" in message
            assert "native API 2" in message
            assert "stale" in message
            assert "rebuild" in message
        else:
            raise AssertionError("a stale native extension must be rejected")
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loaded_native_extension_matches_python_api():
    import robustcov as rc

    if not rc.native_available():
        pytest.skip("native extension is unavailable in this test environment")
    from robustcov import _robustcov_cpp as cpp

    assert cpp.__robustcov_native_api__ == 2


def test_native_free_cmake_option_is_declared():
    cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    assert "option(ROBUSTCOV_BUILD_NATIVE" in cmake
    assert "if(ROBUSTCOV_BUILD_NATIVE)" in cmake


def test_package_smoke_script_compiles_and_has_help():
    script = PROJECT_ROOT / "scripts" / "package_smoke_test.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "expect-native" in result.stdout
