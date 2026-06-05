"""Use case: robust preprocessing for downstream ML.

A classifier is trained on data with contaminated training rows. Robust distances
are used to remove highly suspicious training observations before fitting the
classifier. This is a small smoke test for ML-pipeline usage.

Run:
    python examples/use_case_ml_preprocessing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import robustcov as rc


if __name__ == "__main__":
    try:
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise SystemExit(f"This example requires scikit-learn: {exc}")

    rng = np.random.default_rng(24)
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_redundant=3,
        class_sep=1.0, flip_y=0.02, random_state=24
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=y, random_state=24)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # contaminate a subset of training rows only
    contam = rng.choice(X_train.shape[0], size=45, replace=False)
    X_train_dirty = X_train.copy()
    X_train_dirty[contam, :4] += rng.normal(7.0, 1.2, size=(contam.size, 4))

    raw = LogisticRegression(max_iter=1000).fit(X_train_dirty, y_train)
    raw_acc = accuracy_score(y_test, raw.predict(X_test))

    scatter = rc.FastMCD(quality="fast", contamination=0.10, random_state=0, n_jobs=1).fit(X_train_dirty)
    det = rc.RobustOutlierDetector(estimator=scatter, threshold="empirical", alpha=0.90).fit(X_train_dirty)
    keep = det.labels_ != -1
    clean = LogisticRegression(max_iter=1000).fit(X_train_dirty[keep], y_train[keep])
    clean_acc = accuracy_score(y_test, clean.predict(X_test))

    print("robust preprocessing for downstream ML")
    print(f"raw_training_accuracy_on_test={raw_acc:.3f}")
    print(f"robust_filtered_accuracy_on_test={clean_acc:.3f}")
    print(f"removed_training_rows={int((~keep).sum())}")
    print(f"scatter_radial_kurtosis={scatter.radial_kurtosis_:.3f}")

    outdir = Path("results/use_cases/ml_preprocessing")
    outdir.mkdir(parents=True, exist_ok=True)
    rc.plot_robust_distance_profile(scatter, labels=(~keep).astype(int), output_path=outdir / "distance_profile.png", show=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["raw", "robust filtered"], [raw_acc, clean_acc])
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("Downstream accuracy before/after robust filtering")
    fig.savefig(outdir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved diagnostics to", outdir)
