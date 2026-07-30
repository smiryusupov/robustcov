"""Temporary regression coverage for the FastMCD diagnostic contract."""

import numpy as np
import pytest

import robustcov as rc


pytestmark = [pytest.mark.unit, pytest.mark.native]


def _contaminated_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    X[:20] += 8.0
    return X


def test_fast_mcd_rejects_zero_initial_c_steps():
    with pytest.raises(ValueError, match="initial_c_steps must be positive"):
        rc.FastMCD(initial_c_steps=0).fit(_contaminated_sample())


def test_fast_mcd_diagnostics_describe_selected_solution():
    fitted = rc.FastMCD(
        n_init=12,
        n_best=4,
        initial_c_steps=1,
        max_iter=1,
        random_state=0,
    ).fit(_contaminated_sample())

    assert fitted.n_iter_ == 1
    assert fitted.converged_ is False
    assert fitted.raw_objective_value_ == pytest.approx(
        np.linalg.slogdet(fitted.raw_covariance_)[1]
    )
    assert fitted.objective_value_ == pytest.approx(
        np.linalg.slogdet(fitted.covariance_)[1]
    )
    assert np.isfinite(fitted.c_step_objective_value_)
