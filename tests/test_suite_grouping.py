from __future__ import annotations

from pathlib import Path

from suite_groups import PRIMARY_MODULES, SLOW_MODULES, primary_group


ROOT = Path(__file__).resolve().parent


def test_specialist_suite_groups_are_disjoint_and_reference_real_modules():
    ownership: dict[str, str] = {}
    for group, modules in PRIMARY_MODULES.items():
        for name in modules:
            assert name not in ownership, (name, ownership[name], group)
            ownership[name] = group
            assert (ROOT / name).is_file(), (group, name)


def test_every_test_module_has_one_primary_group():
    for path in ROOT.glob("test_*.py"):
        assert primary_group(path) in {
            "unit",
            "integration",
            "statistical",
            "benchmark",
            "native",
            "packaging",
        }


def test_slow_modules_exist():
    assert all((ROOT / name).is_file() for name in SLOW_MODULES)
