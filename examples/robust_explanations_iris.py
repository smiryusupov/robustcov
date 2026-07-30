"""Robust SHAP and LIME references on a contaminated Iris example.

The predictive model is trained on clean data. Only the explainer reference
matrix is contaminated, isolating the question these adapters address: how much
can a few bad reference rows move a local explanation?

Install the optional integrations with::

    python -m pip install "robustcov[explain]"
"""

from __future__ import annotations

import numpy as np

import robustcov as rc


def _weight_vector(explanation, label: int, n_features: int) -> np.ndarray:
    weights = np.zeros(n_features)
    for feature, weight in explanation.local_exp[label]:
        weights[int(feature)] = float(weight)
    return weights


def main() -> None:
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

    print(
        "contaminated rows retained by robust support:",
        int(reference.support_[contaminated_rows].sum()),
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
        print(f"LIME weight drift, contaminated empirical: {empirical_drift:.3f}")
        print(f"LIME weight drift, robust reference:       {robust_drift:.3f}")


if __name__ == "__main__":
    main()
