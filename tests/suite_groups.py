"""Central pytest suite-group classification.

Every test receives one primary marker. Files not listed in a specialist group
belong to the fast unit suite. ``slow`` is an additional orthogonal marker.
"""

from __future__ import annotations

from pathlib import Path


PRIMARY_MODULES: dict[str, frozenset[str]] = {
    "integration": frozenset(
        {
            "test_estimator_protocol.py",
            "test_external_datasets.py",
            "test_external_snapshot_publisher.py",
            "test_sklearn_interoperability.py",
            "test_public_api_contract.py",
        }
    ),
    "statistical": frozenset(
        {
            "test_statistical_hardening.py",
            "test_structured_statistical_hardening.py",
            "test_precision_geometry_monitoring_hardening.py",
            "test_m_estimator_optimizations.py",
            "test_dpd_pca.py",
        }
    ),
    "benchmark": frozenset(
        {
            "test_benchmark_scope_coverage.py",
            "test_distributionally_robust_pca_benchmark.py",
            "test_latent_structure_benchmarks.py",
            "test_method_comparison_benchmark.py",
        }
    ),
    "native": frozenset({"test_native_kernels.py"}),
    "packaging": frozenset(
        {
            "test_packaging_resilience.py",
            "test_release_readiness.py",
            "test_release_rehearsal.py",
        }
    ),
}

SLOW_MODULES = frozenset(
    {
        "test_dpd_pca.py",
        "test_m_estimator_optimizations.py",
        "test_precision_geometry_monitoring_hardening.py",
        "test_statistical_hardening.py",
        "test_structured_statistical_hardening.py",
        "test_distributionally_robust_pca_benchmark.py",
        "test_latent_structure_benchmarks.py",
        "test_method_comparison_benchmark.py",
    }
)


def primary_group(path: str | Path) -> str:
    """Return the one primary suite group for a test module."""

    name = Path(path).name
    matches = [group for group, modules in PRIMARY_MODULES.items() if name in modules]
    if len(matches) > 1:
        raise RuntimeError(f"test module {name!r} belongs to multiple groups: {matches}")
    return matches[0] if matches else "unit"
