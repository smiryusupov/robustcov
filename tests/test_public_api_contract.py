from __future__ import annotations

import importlib.resources
import json

import robustcov as rc
import robustcov.experimental as experimental


TIERS = ("stable_top_level", "provisional_top_level", "experimental_top_level")


def _manifest() -> dict[str, object]:
    text = (
        importlib.resources.files("robustcov")
        .joinpath("_public_api.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def test_every_top_level_export_is_classified_exactly_once():
    manifest = _manifest()
    classified = [name for tier in TIERS for name in manifest[tier]]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(rc.__all__)


def test_experimental_namespace_matches_manifest():
    manifest = _manifest()
    assert set(manifest["experimental_namespace"]) == set(experimental.__all__)


def test_manifest_version_matches_runtime():
    assert _manifest()["package_version"] == rc.__version__


def test_manifest_symbols_are_bound_in_the_declared_namespace():
    manifest = _manifest()
    for tier in TIERS:
        for name in manifest[tier]:
            assert hasattr(rc, name), (tier, name)
    for name in manifest["experimental_namespace"]:
        assert hasattr(experimental, name), name
