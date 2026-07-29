#!/usr/bin/env python3
"""Install built robustcov artifacts into isolated environments and smoke-test them."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _smoke_code(expect_native: str) -> str:
    return f"""
import numpy as np
import robustcov as rc

print('version=', rc.__version__)
print('module=', rc.__file__)
print('native=', rc.native_available())
print('openmp=', rc.has_openmp())

expected = {expect_native!r}
if expected == 'yes':
    assert rc.native_available(), 'expected the native extension to be available'
elif expected == 'no':
    assert not rc.native_available(), 'expected a native-free installation'

X = np.random.default_rng(0).normal(size=(40, 4))
model = rc.RegularizedCauchy(alpha=0.1, max_iter=20).fit(X)
assert model.covariance_.shape == (4, 4)
assert np.all(np.isfinite(model.covariance_))

if rc.native_available():
    native_model = rc.FastMCD(n_init=5, random_state=0).fit(X)
    assert native_model.covariance_.shape == (4, 4)
else:
    assert rc.get_num_threads() == 1
    rc.set_num_threads(1)
    try:
        rc.FastMCD(n_init=5, random_state=0).fit(X)
    except RuntimeError as exc:
        assert 'requires the robustcov native extension' in str(exc)
    else:
        raise AssertionError('FastMCD should fail clearly without the extension')
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--expect-native",
        choices=("any", "yes", "no"),
        default="any",
        help="assert whether the installed artifact contains the native extension",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="pass --no-deps to pip (useful with a prepared CI environment)",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="pass --no-build-isolation when installing source distributions",
    )
    parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="create test environments with access to the caller's installed packages",
    )
    args = parser.parse_args()

    artifacts = [artifact.resolve() for artifact in args.artifacts]
    for artifact in artifacts:
        if not artifact.is_file():
            parser.error(f"artifact does not exist: {artifact}")

    for artifact in artifacts:
        with tempfile.TemporaryDirectory(prefix="robustcov-package-smoke-") as tmp:
            root = Path(tmp)
            venv = root / "venv"
            venv_command = [sys.executable, "-m", "venv"]
            if args.system_site_packages:
                venv_command.append("--system-site-packages")
            venv_command.append(str(venv))
            _run(venv_command)
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            install = [str(python), "-m", "pip", "install"]
            if args.no_deps:
                install.append("--no-deps")
            if args.no_build_isolation:
                install.append("--no-build-isolation")
            install.append(str(artifact))
            _run(install, cwd=root)
            _run([str(python), "-c", _smoke_code(args.expect_native)], cwd=root)
            print(f"PASS: {artifact.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
