"""Robust SHAP and LIME references on a contaminated Iris example.

The predictive model is trained on clean data. Only the explainer reference
matrix is contaminated, isolating the question these adapters address: how much
can a few bad reference rows move a local explanation?

Install the optional integrations with::

    python -m pip install "robustcov[explain]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import robustcov as rc


def _weight_vector(explanation, label: int, n_features: int) -> np.ndarray:
    weights = np.zeros(n_features)
    for feature, weight in explanation.local_exp[label]:
        weights[int(feature)] = float(weight)
    return weights


def _write_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    empirical: list[float] = []
    robust: list[float] = []
    for name in ("shap", "lime"):
        result = report[name]
        if not result["available"]:
            continue
        labels.append(name.upper())
        empirical.append(float(result["contaminated_empirical_drift"]))
        robust.append(float(result["robust_reference_drift"]))

    if not labels:
        return

    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar(positions - width / 2, empirical, width, label="Contaminated empirical")
    axis.bar(positions + width / 2, robust, width, label="Robust reference")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Total absolute attribution drift")
    axis.set_title("Explanation drift from clean reference")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_experiment() -> dict[str, Any]:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    keep = iris.target < 2  # binary problem keeps SHAP's linear example compact
    X = iris.data[keep]
    y = iris.target[keep]
    feature_names = list(iris.feature_names)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, random_state=0).fit(X_train, y_train)

    rng = np.random.default_rng(17)
    contaminated = X_train.copy()
    contaminated_rows = rng.choice(len(contaminated), size=12, replace=False)
    contaminated[contaminated_rows] += rng.normal(
        loc=np.array([9.0, -8.0, 10.0, -9.0]),
        scale=0.8,
        size=(len(contaminated_rows), X_train.shape[1]),
    )

    reference = rc.RobustExplanationReference(max_samples=50).fit(contaminated)
    query = X_test[[0]]
    retained = int(reference.support_[contaminated_rows].sum())

    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "sklearn Iris classes 0 and 1",
        "model": "standardized logistic regression",
        "n_training_rows": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "contaminated_rows": int(len(contaminated_rows)),
        "contaminated_rows_retained": retained,
        "robust_support_size": int(reference.support_.sum()),
        "shap": {"available": False},
        "lime": {"available": False},
    }

    print(
        "contaminated rows retained by robust support:",
        retained,
        "of",
        len(contaminated_rows),
    )

    try:
        import shap
    except ImportError:
        print("SHAP not installed; skipping SHAP comparison")
    else:

        def linear_values(mean, covariance):
            masker = shap.maskers.Impute(
                {"mean": mean, "cov": covariance}, method="linear"
            )
            np.random.seed(123)
            explainer = shap.LinearExplainer(model, masker, nsamples=500)
            return np.asarray(explainer(query).values)

        clean_values = linear_values(
            np.mean(X_train, axis=0), np.cov(X_train, rowvar=False)
        )
        contaminated_values = linear_values(
            np.mean(contaminated, axis=0), np.cov(contaminated, rowvar=False)
        )
        np.random.seed(123)
        robust_explainer = rc.make_shap_explainer(
            model,
            reference,
            correlation_dependent=True,
            nsamples=500,
        )
        robust_values = np.asarray(robust_explainer(query).values)

        empirical_drift = float(np.abs(contaminated_values - clean_values).sum())
        robust_drift = float(np.abs(robust_values - clean_values).sum())
        report["shap"] = {
            "available": True,
            "contaminated_empirical_drift": empirical_drift,
            "robust_reference_drift": robust_drift,
            "drift_ratio": robust_drift / empirical_drift,
        }
        print(f"SHAP attribution drift, contaminated empirical: {empirical_drift:.3f}")
        print(f"SHAP attribution drift, robust reference:       {robust_drift:.3f}")

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("LIME not installed; skipping LIME comparison")
    else:
        lime_kwargs = dict(
            mode="classification",
            class_names=list(iris.target_names[:2]),
            feature_names=feature_names,
            discretize_continuous=False,
            random_state=123,
        )
        clean_lime = LimeTabularExplainer(X_train, **lime_kwargs)
        contaminated_lime = LimeTabularExplainer(contaminated, **lime_kwargs)
        robust_lime = rc.make_lime_tabular_explainer(reference, **lime_kwargs)
        label = int(model.predict(query)[0])
        explain_kwargs = dict(
            labels=(label,), num_features=X_train.shape[1], num_samples=2000
        )

        clean_exp = clean_lime.explain_instance(
            query[0], model.predict_proba, **explain_kwargs
        )
        contaminated_exp = contaminated_lime.explain_instance(
            query[0], model.predict_proba, **explain_kwargs
        )
        robust_exp = robust_lime.explain_instance(
            query[0], model.predict_proba, **explain_kwargs
        )

        clean_weights = _weight_vector(clean_exp, label, X_train.shape[1])
        contaminated_weights = _weight_vector(
            contaminated_exp, label, X_train.shape[1]
        )
        robust_weights = _weight_vector(robust_exp, label, X_train.shape[1])
        empirical_drift = float(np.abs(contaminated_weights - clean_weights).sum())
        robust_drift = float(np.abs(robust_weights - clean_weights).sum())
        report["lime"] = {
            "available": True,
            "contaminated_empirical_drift": empirical_drift,
            "robust_reference_drift": robust_drift,
            "drift_ratio": robust_drift / empirical_drift,
        }
        print(f"LIME weight drift, contaminated empirical: {empirical_drift:.3f}")
        print(f"LIME weight drift, robust reference:       {robust_drift:.3f}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()

    report = run_experiment()
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.plot_output is not None:
        _write_plot(report, args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
