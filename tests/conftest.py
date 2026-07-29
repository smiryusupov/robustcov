"""Pytest configuration and release-suite grouping."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from suite_groups import SLOW_MODULES, primary_group


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source-tree scripts launched by tests must import the repository package just
# as they would after installation. Centralizing this here removes the recurring
# need for individual tests or contributors to export PYTHONPATH manually.
_pythonpath = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
if str(PROJECT_ROOT) not in _pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), *_pythonpath])

try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
else:
    # Use a non-interactive backend in CI/headless environments.
    # Plotting tests use pytest.importorskip when Matplotlib is absent.
    matplotlib.use("Agg", force=True)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach one primary release-suite marker and an optional slow marker."""

    for item in items:
        module_name = Path(str(item.path)).name
        item.add_marker(getattr(pytest.mark, primary_group(module_name)))
        if module_name in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
