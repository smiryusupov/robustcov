from __future__ import annotations

import numpy as np
import pytest

import robustcov as rc


def _contaminated_numeric_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=(120, 4))
    contaminated = clean.copy()
    contaminated[-12:] += np.array([14.0, -11.0, 9.0, 12.0])
    return clean, contaminated


def test_robust_explanation_reference_filters_contaminated_rows():
    clean, contaminated = _contaminated_numeric_data()
    reference = rc.RobustExplanationReference(max_samples=30).fit(contaminated)

    assert reference.background_.shape == (30, 4)
    assert np.all(reference.background_indices_ < 108)
    assert np.linalg.norm(reference.location_ - np.mean(clean, axis=0)) < 0.8
    np.testing.assert_allclose(reference.mean_covariance()[0], reference.location_)
    assert reference.support_fraction_ < 1.0


def test_shap_adapter_uses_robust_covariance_for_linear_explanations():
    pytest.importorskip("shap")
    linear_model = pytest.importorskip("sklearn.linear_model")

    clean, contaminated = _contaminated_numeric_data(seed=4)
    target = clean @ np.array([1.2, -0.7, 0.4, 0.9])
    model = linear_model.LinearRegression().fit(clean, target)
    reference = rc.RobustExplanationReference(max_samples=40).fit(contaminated)

    explainer = rc.make_shap_explainer(
        model,
        reference,
        correlation_dependent=True,
        nsamples=100,
    )
    values = explainer(clean[:2])

    assert values.values.shape == (2, 4)
    assert np.isfinite(values.values).all()
    np.testing.assert_allclose(explainer.mean, reference.location_)
    np.testing.assert_allclose(explainer.cov, reference.covariance_)
    assert explainer.robust_reference_ is reference


def test_lime_adapter_generates_a_local_explanation():
    pytest.importorskip("lime.lime_tabular")
    linear_model = pytest.importorskip("sklearn.linear_model")

    clean, contaminated = _contaminated_numeric_data(seed=9)
    labels = (clean[:, 0] - 0.8 * clean[:, 1] + 0.4 * clean[:, 2] > 0).astype(int)
    model = linear_model.LogisticRegression().fit(clean, labels)
    explainer = rc.make_lime_tabular_explainer(
        contaminated,
        mode="classification",
        class_names=["negative", "positive"],
        feature_names=["x0", "x1", "x2", "x3"],
        max_samples=40,
    )

    explanation = explainer.explain_instance(
        clean[0],
        model.predict_proba,
        labels=(1,),
        num_features=4,
        num_samples=250,
    )

    assert 1 in explanation.local_exp
    assert len(explanation.local_exp[1]) == 4
    assert explainer.scaled_precision_.shape == (4, 4)
