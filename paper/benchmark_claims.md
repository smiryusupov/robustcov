# Benchmark claims for future JOSS submission

This file tracks quantitative claims that may appear in the JOSS paper. Each
claim has a source page, source artifact, and reproduction command or source
file before it is used in `paper.md`.

## Claim 1: Heavy-tail scatter benchmark

**Paper-ready claim**

Across 27 small-sample heavy-tail benchmark settings, the `robustcov Cauchy`
estimator achieved median relative Frobenius error `0.5994`, compared with
`2.1739` for scikit-learn `MinCovDet`.

**Purpose in paper**

Supports the statistical gap: mainstream robust covariance tooling is limited
for small-sample, heavy-tailed settings.

**Source docs**

    docs/benchmarks/small_sample_heavy_tail.rst

**Source script**

    benchmarks/small_sample_heavy_tail.py

**Source artifact**

    docs/_static/benchmarks/small_sample_summary.csv

**Exact rows**

    robustcov Cauchy,27,0,0.7407,1.4074,0.5994,0.7625,0.013307
    sklearn MinCovDet,27,0,0.1111,6.2963,2.1739,15.6922,0.024392

**Status**

Verified for paper.

---

## Claim 2: Win rate / estimator ranking

**Paper-ready claim**

In the same 27-setting heavy-tail benchmark, `robustcov Cauchy` had win rate
`0.7407`, mean rank `1.4074`, and no failures.

**Purpose in paper**

Supports the claim that the package is useful beyond one favorable example.

**Source docs**

    docs/benchmarks/small_sample_heavy_tail.rst

**Source script**

    benchmarks/small_sample_heavy_tail.py

**Source artifact**

    docs/_static/benchmarks/small_sample_summary.csv

**Exact row**

    robustcov Cauchy,27,0,0.7407,1.4074,0.5994,0.7625,0.013307

**Status**

Verified for paper.

---

## Claim 3: C++ / pybind speed comparison

**Paper-ready claim**

In the documented speed benchmark, `robustcov FastMCD` had median runtime
`0.023761` seconds, compared with `0.191902` seconds for scikit-learn
`MinCovDet`, corresponding to an approximately `8.08x` speedup.

**Purpose in paper**

Supports the software-engineering contribution.

**Source docs**

    docs/benchmarks/speed_comparison.rst
    docs/benchmarks/openmp_scaling.rst

**Source script**

    benchmarks/speed_estimators.py
    benchmarks/openmp_scaling.py

**Source artifact**

    docs/_static/benchmarks/speed.csv

**Exact rows**

    robustcov FastMCD,0.023761,0.023498,0.024421
    sklearn MinCovDet,0.191902,0.190605,0.193694

**Calculation**

    0.191902 / 0.023761 = 8.0763

**Status**

Verified for paper.

---

## Claim 4: Robust GP / kernel input metric

**Paper-ready claim**

In the robust GP input-metric example, RMSE improved from `0.3566` with the
empirical input covariance metric to `0.1813` with the robust input covariance
metric.

**Purpose in paper**

Supports the kernel/GP gap: this is input-space robustness, not output
likelihood robustness.

**Source docs**

    docs/gallery/gp_robust_input_metric.rst

**Source script**

    examples/gp_robust_input_metric.py

**Source artifact**

    docs/_static/gallery/gp_robust_input_metric/output.txt

**Exact output**

    gp_empirical_input_covariance_rmse=0.3566
    gp_robust_input_covariance_rmse=0.1813

**Status**

Verified for paper.

---

## Claim 5: Hard contamination honesty statement

**Paper-ready claim**

The hard-contamination benchmark reports both favorable and unfavorable
scenarios. For example, in the mean-shift scenario scikit-learn `MinCovDet`
has lower relative Frobenius error than `robustcov FastMCD`, while in the
heavy-tail-inlier scenario `robustcov FastMCD` has lower relative Frobenius
error.

**Purpose in paper**

Builds reviewer trust by showing that the package does not overclaim.

**Source docs**

    docs/benchmarks/hard_contamination.rst

**Source script**

    benchmarks/hard_contamination_scenarios.py

**Source artifact**

    docs/_static/benchmarks/hard_scenarios.csv

**Exact rows**

    mean_shift,0.2,robustcov FastMCD,0.1352,0.046352,1.0000,0.0000,786,142.4016
    mean_shift,0.2,sklearn MinCovDet,0.1107,0.123946,1.0000,0.0000,786,

    heavy_tail_inliers,0.2,robustcov FastMCD,0.0886,0.046087,,,,7.7035
    heavy_tail_inliers,0.2,sklearn MinCovDet,0.1205,0.114427,,,,

**Status**

Verified for paper.
