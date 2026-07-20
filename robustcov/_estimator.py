# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Small scikit-learn-compatible estimator protocol without a hard dependency.

The public estimators only need a small part of ``sklearn.base.BaseEstimator``:
parameter introspection, nested parameter updates, and a useful representation.
Keeping that protocol local lets robustcov remain usable without scikit-learn while
still supporting ``sklearn.base.clone``, pipelines, and parameter searches whenever
scikit-learn is installed.
"""

from __future__ import annotations

import inspect
from pprint import pformat


class EstimatorMixin:
    """Implement the estimator parameter protocol expected by scikit-learn."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        signature = inspect.signature(init)
        parameters = []
        for parameter in signature.parameters.values():
            if parameter.name == "self" or parameter.kind == parameter.VAR_KEYWORD:
                continue
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} estimators must not use *args in __init__"
                )
            parameters.append(parameter.name)
        return sorted(parameters)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return constructor parameters, including nested estimator parameters."""
        out: dict[str, object] = {}
        for key in self._get_param_names():
            if not hasattr(self, key):
                raise AttributeError(
                    f"{type(self).__name__}.__init__ exposes {key!r} but the "
                    "instance does not store an attribute with that name"
                )
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                for nested_key, nested_value in value.get_params().items():
                    out[f"{key}__{nested_key}"] = nested_value
            out[key] = value
        return out

    def set_params(self, **params):
        """Set constructor parameters using scikit-learn's ``name__child`` syntax."""
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested: dict[str, dict[str, object]] = {}
        for key, value in params.items():
            root, delimiter, child = key.partition("__")
            if root not in valid:
                valid_names = ", ".join(sorted(self.get_params(deep=False)))
                raise ValueError(
                    f"Invalid parameter {root!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {valid_names}."
                )
            if delimiter:
                nested.setdefault(root, {})[child] = value
            else:
                setattr(self, root, value)
        for root, child_params in nested.items():
            child = getattr(self, root)
            if not hasattr(child, "set_params"):
                raise ValueError(f"Parameter {root!r} does not support nested parameters")
            child.set_params(**child_params)
        return self

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        rendered = ", ".join(f"{name}={pformat(value)}" for name, value in params.items())
        return f"{type(self).__name__}({rendered})"
