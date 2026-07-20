"""Regression checks for the task-oriented public FAQ."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAQ = ROOT / "docs" / "faq.rst"


def test_faq_is_task_oriented_and_current():
    text = FAQ.read_text(encoding="utf-8")

    for heading in (
        "Choosing a method",
        "PCA, decomposition, and latent structure",
        "Monitoring and anomaly alerts",
        "Data structure and missingness",
        "API, performance, and reproducibility",
    ):
        assert heading in text

    for public_name in (
        "FastMCD",
        "MRCD",
        "CellMCD",
        "CellRCov",
        "PrincipalComponentPursuit",
        "RobustPCA",
        "CellPCA",
        "ConformalAlertCalibrator",
        "RobustSubspaceMonitor",
        "OnlineRobustSubspaceTracker",
        "SpectralFilteringCovariance",
    ):
        assert f"``{public_name}``" in text


def test_faq_avoids_obsolete_or_universal_ranking_claims():
    text = FAQ.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "why not prioritize mve" not in lowered
    assert "strongest method" not in lowered
    assert "strongest method in the package" not in lowered
    assert "universal ranking" in lowered
    assert "scenario-specific evidence" in lowered


def test_faq_links_to_authoritative_detail_pages():
    text = FAQ.read_text(encoding="utf-8")
    for target in (
        "estimator_guide",
        "workflows",
        "method_comparison",
        "api_stability",
        "use_case_gallery",
        "benchmark_gallery",
        "external_data",
    ):
        assert f":doc:`{target}`" in text
