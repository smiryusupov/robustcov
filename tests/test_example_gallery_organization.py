"""Checks for method-oriented example discovery and new source-separation scripts."""

from __future__ import annotations

import importlib.util
import subprocess
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"


def _load_runner_module():
    path = EXAMPLES / "run_use_case_gallery.py"
    spec = importlib.util.spec_from_file_location("robustcov_example_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_groups_reference_existing_scripts():
    runner = _load_runner_module()
    assert {"quick", "ica", "pca", "robust", "monitoring"} <= set(runner.GROUPS)
    assert "principal_component_pursuit.py" in runner.GROUPS["pca"]
    assert "distributionally_robust_pca.py" in runner.GROUPS["pca"]
    assert "distributionally_robust_pca_drift_monitoring.py" in runner.GROUPS["pca"]
    assert "distributionally_robust_pca_drift_monitoring.py" in runner.GROUPS["monitoring"]
    assert "online_robust_subspace_tracking.py" in runner.GROUPS["pca"]
    assert "online_robust_subspace_tracking.py" in runner.GROUPS["monitoring"]
    assert "adversarial_covariance_filtering.py" in runner.GROUPS["robust"]
    for scripts in runner.GROUPS.values():
        for script in scripts:
            assert (EXAMPLES / script).is_file(), script


def test_method_gallery_pages_and_detailed_examples_exist():
    expected = [
        DOCS / "gallery_methods" / "ica_source_separation.rst",
        DOCS / "gallery_methods" / "pca_factor_models.rst",
        DOCS / "gallery_methods" / "robust_estimators.rst",
        DOCS / "gallery_methods" / "anomaly_monitoring.rst",
        DOCS / "gallery" / "ica_two_scatter.rst",
        DOCS / "gallery" / "sobi_source_separation.rst",
        DOCS / "gallery" / "robust_factor_model.rst",
        DOCS / "gallery" / "principal_component_pursuit.rst",
        DOCS / "gallery" / "distributionally_robust_pca.rst",
        DOCS / "gallery" / "distributionally_robust_pca_drift_monitoring.rst",
        DOCS / "gallery" / "online_robust_subspace_tracking.rst",
        DOCS / "gallery" / "adversarial_covariance_filtering.rst",
    ]
    for path in expected:
        assert path.is_file(), path

    expected_images = [
        DOCS / "_static" / "gallery" / "ica_two_scatter" / "source_recovery.png",
        DOCS / "_static" / "gallery" / "ica_two_scatter" / "mixture_and_sources.png",
        DOCS / "_static" / "gallery" / "sobi_source_separation" / "source_recovery.png",
        DOCS / "_static" / "gallery" / "sobi_source_separation" / "lag_signatures.png",
        DOCS / "_static" / "gallery" / "sobi_source_separation" / "mdi_comparison.png",
        DOCS / "_static" / "gallery" / "robust_factor_model" / "loading_recovery.png",
        DOCS / "_static" / "gallery" / "robust_factor_model" / "factor_scores.png",
        DOCS / "_static" / "gallery" / "robust_factor_model" / "factor_selection.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca" / "target_risk.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca" / "subspace_allocation.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca" / "ambiguity_path.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca_drift_monitoring" / "drift_timeline.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca_drift_monitoring" / "alert_rates.png",
        DOCS / "_static" / "gallery" / "distributionally_robust_pca_drift_monitoring" / "feature_contributions.png",
    ]
    for path in expected_images:
        assert path.is_file() and path.stat().st_size > 10_000, path


@pytest.mark.parametrize(
    ("script", "expected_text"),
    [
        ("principal_component_pursuit.py", "Principal Component Pursuit,"),
        ("ica_two_scatter.py", "Minimum-distance index:"),
        ("sobi_source_separation.py", "Robust SOBI MDI:"),
        ("robust_factor_model.py", "Selected factor count:"),
        ("distributionally_robust_pca.py", "selected_path_gamma,"),
        ("distributionally_robust_pca_drift_monitoring.py", "worst_case_risk_role,model_diagnostic_not_alert_threshold"),
    ],
)
def test_new_method_examples_run(script: str, expected_text: str, tmp_path):
    pytest.importorskip("matplotlib")
    outdir = tmp_path / Path(script).stem
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(ROOT)
    script_path = EXAMPLES / script
    script_args = [str(script_path), "--outdir", str(outdir)]
    import_paths = [str(ROOT), *[path for path in sys.path if path and "site-packages" in path]]
    bootstrap = (
        "import runpy, sys; "
        f"sys.path[:0] = {import_paths!r}; "
        f"sys.argv = {script_args!r}; "
        f"runpy.run_path({str(script_path)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", bootstrap],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_text in result.stdout
    expected_images = {
        "principal_component_pursuit.py": {"decomposition.png", "convergence.png"},
        "ica_two_scatter.py": {"source_recovery.png", "mixture_and_sources.png"},
        "sobi_source_separation.py": {"source_recovery.png", "lag_signatures.png", "mdi_comparison.png"},
        "robust_factor_model.py": {"loading_recovery.png", "factor_scores.png", "factor_selection.png"},
        "distributionally_robust_pca.py": {"target_risk.png", "subspace_allocation.png", "ambiguity_path.png"},
        "distributionally_robust_pca_drift_monitoring.py": {"drift_timeline.png", "alert_rates.png", "feature_contributions.png"},
    }[script]
    assert expected_images <= {path.name for path in outdir.glob("*.png")}


def test_example_navigation_stays_category_only():
    """The global sidebar should expose categories, not every example page."""

    gallery_index = (DOCS / "use_case_gallery.rst").read_text(encoding="utf-8")
    assert "All detailed pages" not in gallery_index
    assert "   gallery/" not in gallery_index

    expected_categories = {
        "gallery_methods/ica_source_separation",
        "gallery_methods/pca_factor_models",
        "gallery_methods/robust_estimators",
        "gallery_methods/anomaly_monitoring",
        "gallery_topics/finance_and_risk",
        "gallery_topics/fraud_security_and_networks",
        "gallery_topics/sensors_industrial_quality",
        "gallery_topics/biomedical_images_embeddings",
        "gallery_topics/real_ml_datasets",
        "gallery_topics/ml_preprocessing",
    }
    for category in expected_categories:
        assert f"   {category}" in gallery_index

    for category_dir in (DOCS / "gallery_methods", DOCS / "gallery_topics"):
        for page in category_dir.glob("*.rst"):
            text = page.read_text(encoding="utf-8")
            assert ".. toctree::" not in text, page

    detail_pages = sorted((DOCS / "gallery").glob("*.rst"))
    assert detail_pages
    for page in detail_pages:
        assert page.read_text(encoding="utf-8").startswith(":orphan:\n"), page
