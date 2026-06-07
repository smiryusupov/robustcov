"""Generate static Sphinx gallery assets from runnable use-case examples.

This script runs each gallery example, captures stdout/stderr, and copies the
plots saved under results/use_cases into docs/_static/gallery/<slug>/ so the
Sphinx pages show real outputs instead of only instructions.

Run from the repository root:
    python docs/generate_gallery_assets.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GalleryCase:
    slug: str
    script: str
    result_dir: str | None
    images: tuple[str, ...]


CASES = [
    GalleryCase("finance_risk", "use_case_finance_risk.py", "finance", ("distance_panel.png", "covariance.png")),
    GalleryCase("portfolio_stress", "use_case_portfolio_stress.py", "portfolio_stress", ("distance_panel.png", "covariance.png")),
    GalleryCase("fraud_screening", "use_case_fraud_screening.py", "fraud", ("distance_panel.png", "distance_profile.png")),
    GalleryCase("sensor_anomaly", "use_case_sensor_anomaly.py", "sensor", ("distance_panel.png", "distance_profile.png")),
    GalleryCase("maintenance_monitoring", "use_case_maintenance_monitoring.py", "maintenance", ("distance_panel.png", "time_profile.png")),
    GalleryCase("quality_control", "use_case_quality_control.py", "quality_control", ("support_ellipse.png", "distance_profile.png")),
    GalleryCase("network_traffic", "use_case_network_traffic.py", "network", ("distance_panel.png",)),
    GalleryCase("biomedical_signal", "use_case_biomedical_signal.py", "biomedical_signal", ("distance_profile.png",)),
    GalleryCase("image_feature_anomaly", "use_case_image_feature_anomaly.py", "image_features", ("distance_panel.png",)),
    GalleryCase("text_embedding_outliers", "use_case_text_embedding_outliers.py", "embedding_outliers", ("distance_panel.png",)),
    GalleryCase("breast_cancer_screening", "use_case_breast_cancer_screening.py", "breast_cancer", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("digits_one_class", "use_case_digits_one_class_baselines.py", "digits_one_class", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("wine_class_screening", "use_case_wine_class_screening.py", "wine_class", ("baseline_f1.png", "distance_panel.png", "score_profile.png")),
    GalleryCase("ml_preprocessing", "use_case_ml_preprocessing.py", "ml_preprocessing", ("accuracy_comparison.png", "distance_profile.png")),
    GalleryCase("gp_robust_input_metric", "gp_robust_input_metric.py", "gp_robust_input_metric", ("kernel_comparison.png",)),
    GalleryCase("multimodal_anomaly", "use_case_multimodal_anomaly.py", "multimodal_anomaly", ("cluster_distance_panel.png", "global_distance_profile.png", "metrics.csv")),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "docs" / "_static" / "gallery"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int]] = []

    for case in CASES:
        script_path = root / "examples" / case.script
        case_out = out_root / case.slug
        case_out.mkdir(parents=True, exist_ok=True)
        print(f"running {case.script}")
        try:
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=45,
                env=env,
            )
            returncode = proc.returncode
            output = proc.stdout
            if proc.stderr:
                output += "\n[stderr]\n" + proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            output = (exc.stdout or "")
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            err = exc.stderr or ""
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            output += "\n[timeout] example exceeded 45 seconds\n" + err
        (case_out / "output.txt").write_text(output.strip() + "\n", encoding="utf-8")

        copied = []
        if case.result_dir is not None:
            source_dir = root / "results" / "use_cases" / case.result_dir
            for image in case.images:
                src = source_dir / image
                if src.exists():
                    shutil.copy2(src, case_out / image)
                    copied.append(image)
        (case_out / "manifest.txt").write_text("\n".join(copied) + ("\n" if copied else ""), encoding="utf-8")
        rows.append((case.script, returncode))

    summary = ["script,status"]
    for script, code in rows:
        summary.append(f"{script},{'ok' if code == 0 else f'failed({code})'}")
    (out_root / "summary.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    return 1 if any(code != 0 for _, code in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
