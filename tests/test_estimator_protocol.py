from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pytest
from sklearn.base import clone

import robustcov as rc


def _public_estimator_classes():
    classes = {}
    for name in rc.__all__:
        obj = getattr(rc, name, None)
        if inspect.isclass(obj) and hasattr(obj, "fit"):
            classes.setdefault(obj, name)
    return tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__name__)))


@pytest.mark.parametrize("estimator_class", _public_estimator_classes())
def test_every_public_estimator_supports_clone_and_parameter_protocol(estimator_class):
    estimator = estimator_class()
    shallow = estimator.get_params(deep=False)

    assert set(shallow) == set(estimator_class._get_param_names())
    cloned = clone(estimator)
    assert cloned is not estimator
    assert cloned.get_params(deep=False) == shallow

    with pytest.raises(ValueError, match="Invalid parameter"):
        estimator.set_params(not_a_parameter=1)


@pytest.mark.parametrize(
    ("estimator", "nested_name", "expected"),
    [
        (rc.RobustPCA(estimator=rc.FastMCD()), "estimator__quality", "high"),
        (rc.FeatureGeometry(estimator=rc.FastMCD()), "estimator__quality", "high"),
        (
            rc.ClassConditionalFeatureGeometry(estimator=rc.FastMCD()),
            "estimator__quality",
            "high",
        ),
        (
            rc.RobustSubspaceMonitor(estimator=rc.FastMCD()),
            "estimator__quality",
            "high",
        ),
        (
            rc.SubspaceStability(pca=rc.RobustPCA()),
            "pca__n_components",
            2,
        ),
        (
            rc.RobustGraphicalLasso(scatter_estimator=rc.FastMCD()),
            "scatter_estimator__quality",
            "high",
        ),
        (
            rc.ClusterRobustOutlierDetector(base_estimator=rc.FastMCD()),
            "base_estimator__quality",
            "high",
        ),
    ],
)
def test_nested_estimator_parameters_can_be_updated(estimator, nested_name, expected):
    estimator.set_params(**{nested_name: expected})
    assert estimator.get_params(deep=True)[nested_name] == expected


def test_matrix_mcd_preserves_constructor_parameters_until_fit():
    estimator = rc.MatrixMCD(quality="balanced")
    assert estimator.n_init is None
    assert estimator.n_best is None
    assert estimator.initial_c_steps is None
    assert estimator.max_iter is None
    assert not hasattr(estimator, "backend_")

    rng = np.random.default_rng(123)
    X = rng.normal(size=(18, 2, 2))
    estimator.set_params(n_init=4, n_best=2, initial_c_steps=1, max_iter=3).fit(X)

    assert estimator.n_init == 4
    assert estimator.n_best == 2
    assert estimator.initial_c_steps == 1
    assert estimator.max_iter == 3
    assert estimator.effective_n_init_ == 4
    assert estimator.effective_n_best_ == 2
    assert estimator.effective_initial_c_steps_ == 1
    assert estimator.effective_max_iter_ == 3


def test_auto_scatter_preserves_criterion_alias_as_constructor_parameter():
    estimator = rc.AutoRobustScatter(selection="stability", criterion=None)
    assert estimator.selection == "stability"
    assert estimator.criterion is None

    criterion_estimator = rc.AutoRobustScatter(
        selection="stability", criterion="diagnostic"
    )
    criterion_estimator._validate_parameters()
    assert criterion_estimator.selection == "stability"
    assert criterion_estimator.criterion == "diagnostic"
    assert criterion_estimator.selection_ == "diagnostic"


def test_exported_public_names_exist():
    missing = [name for name in rc.__all__ if not hasattr(rc, name)]
    assert missing == []


def test_documented_quickstart_is_executable():
    project_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        project_root / "docs" / "_snippets" / "quickstart_outlier_detection.py"
    )
    detector = namespace["det"]
    assert detector.labels_.shape == (400,)
    assert int(np.sum(detector.labels_ == -1)) == 30
