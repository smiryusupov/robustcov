"""Run robustcov application use-case examples.

By default this runs a compact, user-facing subset so the command stays quick and
stable on laptops. Use ``--all`` to execute every gallery script.

Run:
    python examples/run_use_case_gallery.py
    python examples/run_use_case_gallery.py --all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

QUICK_SCRIPTS = [
    "use_case_breast_cancer_screening.py",
    "use_case_digits_one_class_baselines.py",
    "use_case_wine_class_screening.py",
    "use_case_finance_risk.py",
    "use_case_fraud_screening.py",
    "use_case_ml_preprocessing.py",
]

ALL_SCRIPTS = QUICK_SCRIPTS + [
    "use_case_sensor_anomaly.py",
    "use_case_quality_control.py",
    "use_case_network_traffic.py",
    "use_case_biomedical_signal.py",
    "use_case_image_feature_anomaly.py",
    "use_case_text_embedding_outliers.py",
    "use_case_portfolio_stress.py",
    "use_case_maintenance_monitoring.py",
    "use_case_multimodal_anomaly.py",
]


def run_script(path: Path, timeout: int):
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print(f"\n$ {sys.executable} {path}")
    try:
        return subprocess.run(
            [sys.executable, str(path)],
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        err += f"\n[timeout] example exceeded {timeout} seconds"
        return subprocess.CompletedProcess([sys.executable, str(path)], 124, out, err)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every gallery example instead of the compact default subset")
    ap.add_argument("--timeout", type=int, default=60, help="per-example timeout in seconds")
    args = ap.parse_args()

    scripts = ALL_SCRIPTS if args.all else QUICK_SCRIPTS
    here = Path(__file__).resolve().parent
    rows = []
    for script in scripts:
        proc = run_script(here / script, args.timeout)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())
        rows.append((script, proc.returncode))

    print("\nuse-case gallery summary")
    for script, code in rows:
        status = "ok" if code == 0 else f"failed({code})"
        print(f"{script},{status}")
    if any(code != 0 for _, code in rows):
        raise SystemExit(1)
