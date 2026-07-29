#!/usr/bin/env python3
"""Smoke-test an installed robustcov package outside the source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version")
    parser.add_argument("--expect-native", choices=("any", "yes", "no"), default="any")
    parser.add_argument(
        "--forbid-root",
        type=Path,
        help="fail if robustcov imports from this source checkout",
    )
    args = parser.parse_args()

    import numpy as np
    import robustcov as rc

    module_path = Path(rc.__file__).resolve()
    if args.forbid_root is not None:
        forbidden = args.forbid_root.resolve()
        try:
            module_path.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"robustcov imported from source checkout: {module_path}")

    if args.expected_version is not None:
        assert rc.__version__ == args.expected_version, (rc.__version__, args.expected_version)

    native = bool(rc.native_available())
    if args.expect_native == "yes":
        assert native, "expected the installed package to contain the native extension"
    elif args.expect_native == "no":
        assert not native, "expected a native-free package"

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8))
    X[0, 0] = 15.0

    scatter = rc.RegularizedCauchy(alpha=0.1, max_iter=40).fit(X)
    assert scatter.covariance_.shape == (8, 8)
    assert np.all(np.isfinite(scatter.covariance_))

    pca = rc.RobustPCA(
        n_components=3,
        estimator=rc.RegularizedCauchy(alpha=0.1, max_iter=40),
    ).fit(X)
    assert pca.components_.shape == (3, 8)

    matrix = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 30))
    matrix[2, 5] += 20.0
    pcp = rc.PrincipalComponentPursuit(max_iter=400, tol=1e-6).fit(matrix)
    assert pcp.low_rank_.shape == matrix.shape
    assert pcp.sparse_.shape == matrix.shape

    calibration = rc.ConformalAlertCalibrator(alpha=0.05).fit(
        np.linspace(0.0, 1.0, 99)
    )
    assert calibration.predict_alerts([0.5, 2.0]).tolist() == [False, True]

    if native:
        native_model = rc.FastMCD(n_init=5, random_state=0).fit(X)
        assert native_model.covariance_.shape == (8, 8)

    print(f"version={rc.__version__}")
    print(f"module={module_path}")
    print(f"native={native}")
    print("installed package smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
