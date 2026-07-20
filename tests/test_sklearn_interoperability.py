from __future__ import annotations

import json
from importlib.resources import files
import inspect

import numpy as np
import pytest

import robustcov as rc


sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def _stable_estimator_classes():
    manifest = json.loads(
        files("robustcov").joinpath("_public_api.json").read_text(encoding="utf-8")
    )
    classes = []
    for name in manifest["stable_top_level"]:
        value = getattr(rc, name)
        if inspect.isclass(value) and hasattr(value, "fit"):
            classes.append(value)
    return sorted(set(classes), key=lambda cls: (cls.__module__, cls.__name__))


@pytest.mark.parametrize("estimator_class", _stable_estimator_classes())
def test_stable_estimators_support_sklearn_clone(estimator_class):
    estimator = estimator_class()
    cloned = clone(estimator)
    assert cloned is not estimator
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)


def test_robust_pca_works_in_pipeline_and_grid_search():
    rng = np.random.default_rng(20260720)
    X = rng.normal(size=(120, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    pipeline = Pipeline(
        [
            ("pca", rc.RobustPCA(n_components=3)),
            ("classifier", LogisticRegression(max_iter=300)),
        ]
    )
    search = GridSearchCV(pipeline, {"pca__n_components": [2, 3]}, cv=2)
    search.fit(X, y)
    assert search.best_estimator_.predict(X).shape == (X.shape[0],)
