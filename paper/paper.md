---
title: 'robustcov: Robust scatter geometry for machine-learning workflows'
tags:
  - Python
  - robust statistics
  - covariance estimation
  - anomaly detection
  - kernel methods
  - information geometry
authors:
  - name: Shohruh Miryusupov
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 2026-07-30
bibliography: paper.bib
---

# Summary

`robustcov` is a Python package that provides a unified pipeline from robust
scatter estimation to SPD geometry and robust kernel/similarity methods for
machine-learning workflows. The package combines a C++/pybind core with Python
estimators, diagnostics, benchmark galleries, and adapters for scikit-learn-style,
GPyTorch-style, SHAP, and LIME workflows. Its central design idea is that robust scatter
estimates should not stop at covariance matrices: they should define robust
precision matrices, Mahalanobis geometry, anomaly scores, whitening transforms,
and input metrics for kernel and similarity methods. `robustcov` is intended for
researchers and practitioners working with contaminated, heavy-tailed, or
leverage-prone data in settings such as finance, biomedical signals, sensor
monitoring, embedding spaces, and robust preprocessing.

# Statement of need

Covariance geometry is used throughout machine learning. Empirical covariance
and Euclidean distance appear directly or indirectly in whitening, Mahalanobis
anomaly scores, Gaussian models, portfolio risk, nearest-neighbor search, kernel
methods, and Gaussian-process input metrics. These tools can be unreliable when
the data contain outliers, leverage points, heavy tails, or small contaminated
subsets. A few abnormal observations can rotate principal directions, inflate
variance estimates, or distort input similarities.

The first gap addressed by `robustcov` is statistical. Mainstream Python
scientific computing provides strong general-purpose tools, and scikit-learn
includes `MinCovDet` as a classical robust covariance estimator
[@pedregosa2011scikit; @rousseeuw1999fast]. However, robust scatter workflows for
small-sample, high-dimensional, and heavy-tailed settings remain less accessible
in a focused Python package. These settings arise in finance, biomedical signal
windows, sensor arrays, and embedding representations. `robustcov` provides
`RegularizedCauchy` and `StudentTScatter` estimators, Tyler-style shape
estimators, regularized Tyler variants, FastMCD, and automatic
estimator-selection utilities [@maronna2019robust; @tyler1987distribution;
@ollila2014regularized].

The second gap is geometric. Kernel and Gaussian-process methods usually assume
an input metric, often Euclidean distance or an automatic-relevance-determination
metric learned by likelihood optimization. When training inputs contain leverage
points or heavy-tailed contamination, this input geometry can be distorted.
`robustcov` treats robust scatter as an input-space geometry layer: robust
scatter gives robust precision, robust precision gives Mahalanobis distance, and
Mahalanobis distance gives robust similarity or kernel geometry. The package
therefore fills a tooling gap between robust covariance estimation and practical
input-robust kernel, Gaussian-process, retrieval, similarity, and explanation-
reference workflows.  For SHAP and LIME, RobustCov supplies a robust background
matrix or robust location/covariance geometry while the upstream packages retain
ownership of the explanation algorithms.

# Software design

`robustcov` follows a modular design. Estimators expose fitted attributes such as
`location_`, `covariance_`, and `precision_`, allowing downstream tools to use a
common interface. FastMCD targets separable contamination; `RegularizedCauchy` and
`StudentTScatter` target small-sample and heavy-tailed settings; `TylerShape`
and `RegularizedTyler` expose scale-invariant shape estimation; and
`AutoRobustScatter` provides a practical default for users who do not want to
choose an estimator manually.

The SPD geometry layer makes the covariance geometry explicit. It provides
matrix logarithms, exponentials, powers, trace and determinant shape
normalization, affine-invariant and log-Euclidean distances, geodesics, and
Tyler fixed-point diagnostics. This layer matters because covariance matrices
live on the SPD cone, where relative eigenvalue changes and variance ratios are
often more meaningful than raw Euclidean differences between matrix entries
[@pennec2006riemannian; @arsigny2007geometric; @wiesel2012geodesic]. The
geometry utilities are intentionally scoped: they support robust covariance
workflows, but do not attempt to turn `robustcov` into a full
information-geometry framework.

The kernel and Gaussian-process adapters follow the same boundary. `robustcov`
owns the input metric; external kernel and GP libraries own inference,
likelihoods, hyperparameter optimization, and posterior prediction. This makes
the robust input geometry reusable without duplicating mature GP infrastructure
[@gardner2018gpytorch].

# Benchmarks and evidence

The documentation includes reproducible benchmark and use-case galleries rather
than embedding large result tables in this paper.  A focused release-evidence
command records the source-tree digest, environment, commands, key results, and
SHA-256 hashes for the numerical areas changed in this release cycle.

In the eight-scenario quick small-sample heavy-tail profile,
`AutoRobustScatter` achieved median relative Frobenius error `0.5907` and win
rate `0.8750`, with its runtime including candidate fitting and selection.
`RegularizedCauchy` was the strongest single estimator across all eight
scenarios with median error `0.5950`.  On the four scenarios where classical
MCD is mathematically applicable, native `FastMCD` had median error `0.6774`
versus `0.8233` for scikit-learn `MinCovDet`.

The statistical validation gate checks unit equivariance, singular inputs, a
contamination curve, and agreement with scikit-learn.  In the committed fixed
reference comparison, FastMCD covariance differs from scikit-learn `MinCovDet`
by relative Frobenius error `0.00535`, with robust-distance correlation
numerically equal to `1.0`.  At 10--30 percent injected row contamination,
FastMCD covariance error remains between `0.126` and `0.135`, while empirical
covariance error ranges from `9.90` to `23.34`.

The explanation-reference example keeps the fitted model and query fixed and
contaminates only the background matrix.  The robust support retains none of
the twelve injected leverage rows.  Total absolute SHAP attribution drift from
the clean reference is `2.528` for the contaminated empirical reference and
`0.388` for the robust reference.  The result demonstrates the purpose of the
adapter without claiming that robust references universally improve every
explanation.

Timing tables are retained as local engineering snapshots rather than portable
constants because compiler, BLAS, processor, and threading choices materially
affect them.  The documentation also reports unfavorable contamination
scenarios so that method choice remains tied to the assumed data-generating
problem.

# Research impact statement

`robustcov` aims to make robust covariance geometry easier to use in applied
research. Many workflows depend on covariance implicitly even when covariance
estimation is not the main research question. By connecting robust scatter
estimators to anomaly scoring, SPD diagnostics, kernel metrics, and embedding
filtering examples, the package gives researchers a reproducible way to test
whether covariance-based conclusions are sensitive to contamination or heavy
tails. The intended impact is not to replace specialized robust-statistics
software, but to provide an accessible Python bridge between robust scatter
methods and modern machine-learning workflows.

# AI usage disclosure

Generative AI tools were used to assist with drafting documentation, examples,
and this manuscript. All generated text and code were reviewed, edited, tested,
and approved by the author. The author is responsible for the final software,
documentation, examples, references, and manuscript content.

# Acknowledgements

The author thanks early users and colleagues who provided feedback on the
package scope, documentation, examples, and JOSS paper framing.

# References
