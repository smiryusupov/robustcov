# Benchmark claims for future JOSS submission

This file tracks quantitative claims that may appear in the JOSS paper. Each
claim should have a source page, source script, and reproduction command before
it is used in `paper.md`.

## Claim 1: Heavy-tail scatter benchmark

**Draft claim**

Across 27 small-sample heavy-tail settings, `RegularizedCauchy` achieved median
Frobenius error around 0.60, compared with around 2.17 for scikit-learn
`MinCovDet`.

**Purpose in paper**

Supports the statistical gap: mainstream robust covariance tooling is limited
for small-sample, heavy-tailed settings.

**Source docs**

TODO: Add benchmark gallery page path.

Example:

    docs/benchmarks/small_sample_heavy_tail.rst

**Source script**

TODO: Add script path.

Example:

    benchmarks/small_sample_heavy_tail.py

**Reproduction command**

TODO.

Example:

    python benchmarks/small_sample_heavy_tail.py

**Exact output / table**

TODO: Paste exact table row or summary here.

**Status**

Not yet verified for paper.

---

## Claim 2: Win rate / estimator ranking

**Draft claim**

`robustcov` estimators won about 74% of benchmark settings in the small-sample
heavy-tail benchmark.

**Purpose in paper**

Supports the claim that the package is useful beyond one favorable example.

**Source docs**

TODO.

**Source script**

TODO.

**Reproduction command**

TODO.

**Exact output / table**

TODO.

**Status**

Not yet verified for paper.

---

## Claim 3: C++ / OpenMP speed comparison

**Draft claim**

The C++/pybind implementation with optional OpenMP showed up to about 8x speedup
over scikit-learn `MinCovDet` in the documented speed benchmark.

**Purpose in paper**

Supports the software-engineering contribution.

**Source docs**

TODO: Add benchmark gallery page path.

Example:

    docs/benchmarks/speed_comparison.rst
    docs/benchmarks/openmp_scaling.rst

**Source script**

TODO.

**Reproduction command**

TODO.

**Exact output / table**

TODO.

**Status**

Not yet verified for paper.

---

## Claim 4: Robust GP / kernel input metric

**Draft claim**

In the robust GP input-metric example, RMSE improved from about 0.357 with the
empirical input metric to about 0.181 with the robust input metric.

**Purpose in paper**

Supports the kernel/GP gap: this is input-space robustness, not output
likelihood robustness.

**Source docs**

TODO: Add gallery page path.

Example:

    docs/gallery/gp_robust_input_metric.rst

**Source script**

TODO.

Example:

    examples/gp_robust_input_metric.py

**Reproduction command**

TODO.

Example:

    python examples/gp_robust_input_metric.py

**Exact output / table**

TODO.

**Status**

Not yet verified for paper.

---

## Claim 5: Hard contamination honesty statement

**Draft claim**

FastMCD does not always outperform scikit-learn `MinCovDet` in mean-shift or
hard-contamination settings; the documentation reports both favorable and
unfavorable scenarios.

**Purpose in paper**

Builds reviewer trust by showing the package does not overclaim.

**Source docs**

TODO: Add benchmark gallery page path.

Example:

    docs/benchmarks/hard_contamination.rst

**Source script**

TODO.

**Reproduction command**

TODO.

**Exact output / table**

TODO.

**Status**

Not yet verified for paper.
