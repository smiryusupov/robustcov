from __future__ import annotations

import re
from pathlib import Path

import robustcov as rc
import robustcov.experimental as experimental
from benchmarks.benchmark_inventory import (
    ALIASES,
    COVERAGE,
    EXPERIMENTAL_ALIASES,
    EXPERIMENTAL_COVERAGE,
)
from robustcov.provenance import (
    EXPERIMENTAL_ESTIMATOR_PROVENANCE_NAMES,
    METHOD_PROVENANCE,
    PUBLIC_ESTIMATOR_PROVENANCE_NAMES,
    REFERENCE_CATALOG,
    STATUS_LABELS,
)
from scripts.generate_method_provenance import render


ROOT = Path(__file__).resolve().parents[1]


def test_every_canonical_public_estimator_has_provenance():
    benchmark_names = {entry.estimator for entry in COVERAGE}
    assert set(PUBLIC_ESTIMATOR_PROVENANCE_NAMES) == benchmark_names
    for name in PUBLIC_ESTIMATOR_PROVENANCE_NAMES:
        assert hasattr(rc, name), name
        entry = rc.get_method_provenance(name)
        assert entry.name == name
        assert entry.status in STATUS_LABELS
        assert entry.summary.strip()
        assert entry.references
        assert entry.robustcov_contribution.strip()
        assert entry.implementation_notes.strip()




def test_every_experimental_estimator_has_provenance_and_benchmark_ownership():
    benchmark_names = {entry.estimator for entry in EXPERIMENTAL_COVERAGE}
    assert set(EXPERIMENTAL_ESTIMATOR_PROVENANCE_NAMES) == benchmark_names
    for name in EXPERIMENTAL_ESTIMATOR_PROVENANCE_NAMES:
        assert hasattr(experimental, name), name
        value = getattr(experimental, name)
        entry = rc.get_method_provenance(value)
        assert entry.name == name
        assert entry.status in STATUS_LABELS
        assert value.method_provenance is entry
        assert entry.references
        assert entry.robustcov_contribution.strip()
        assert entry.implementation_notes.strip()

    for alias, canonical in EXPERIMENTAL_ALIASES.items():
        assert hasattr(experimental, alias), alias
        assert rc.get_method_provenance(getattr(experimental, alias)) is METHOD_PROVENANCE[canonical]

def test_public_aliases_resolve_to_the_canonical_provenance_entry():
    for alias, canonical in ALIASES.items():
        assert hasattr(rc, alias), alias
        assert rc.get_method_provenance(alias) is METHOD_PROVENANCE[canonical]
        assert getattr(rc, alias).canonical_method_name == canonical
        assert alias in METHOD_PROVENANCE[canonical].aliases


def test_runtime_classes_and_algorithms_expose_provenance_attributes():
    names = set(PUBLIC_ESTIMATOR_PROVENANCE_NAMES)
    names.update({"joint_diagonalize_symmetric", "minimum_distance_index", "amari_index"})
    for name in names:
        value = getattr(rc, name)
        entry = rc.get_method_provenance(value)
        assert value.method_provenance is entry
        assert value.method_status == entry.status
        assert value.method_references == entry.references
        assert value.robustcov_contribution == entry.robustcov_contribution
        assert value.implementation_notes == entry.implementation_notes


def test_all_reference_keys_exist_in_catalog_and_bibtex():
    used = {key for entry in METHOD_PROVENANCE.values() for key in entry.references}
    assert used <= set(REFERENCE_CATALOG)

    bib_text = (ROOT / "docs" / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    assert used <= bib_keys

    for key in used:
        reference = REFERENCE_CATALOG[key]
        assert reference.key == key
        assert reference.short.strip()
        assert reference.citation.strip()
        assert reference.url.startswith("https://")


def test_generated_provenance_document_is_current():
    generated = ROOT / "docs" / "_generated" / "method_provenance.inc"
    assert generated.read_text(encoding="utf-8") == render()


def test_method_provenance_and_contributor_rules_scope_public_claims():
    assert all(entry.status != "original_method" for entry in METHOD_PROVENANCE.values())
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Keep benchmark claims scoped" in contributing
    assert "robustcov/_public_api.json" in contributing
    assert not (ROOT / "docs" / "project_contributions.rst").exists()


def test_citation_and_documentation_point_users_to_method_references():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "primary methodological" in citation
    assert "Methods, provenance, and" in citation

    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    assert "methods_and_references" in index
    assert "project_contributions" not in index
