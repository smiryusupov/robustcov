Project contributions and claims policy
=======================================

The project aims to make robust statistical methods practical in modern Python
workflows.  Its contribution is broader than translating formulas into code,
but the documentation deliberately avoids attributing established methodology
to the package.

What robustcov contributes
--------------------------

The current package makes contributions in four areas:

**Unified software design**
   A common estimator protocol connects robust covariance and shape estimation,
   PCA, cellwise and matrix/tensor methods, sparse precision estimation, ICA,
   SOBI, factor models, anomaly detection, feature geometry, and monitoring.

**Numerical engineering**
   The project provides independently written Python and C++ implementations,
   OpenMP kernels, workload-aware backend routing, scale-relative numerical
   safeguards, native/Python compatibility checks, optional native-free builds,
   and reproducible performance gates.

**Package-specific compositions**
   Classes such as ``RobustSOBI``, ``RobustPCA``, ``AutoRobustScatter``,
   ``RobustGraphicalLasso``, experimental ``DistributionallyRobustPCA``, and the feature-geometry/monitoring workflows
   combine established components behind a tested API.  These are described as
   robustcov composites, not as inventions of their component methods.

**Evidence and reproducibility**
   Transformation tests, contamination studies, reference comparisons,
   complete-fit acceleration gates, benchmark ownership, runnable examples, and
   distribution smoke tests are maintained with the implementation.

What robustcov does not currently claim
---------------------------------------

* No current public estimator is labelled as an original statistical method.
* ``RobustPCA`` is not the complete ROBPCA algorithm; it is PCA driven by a
  configurable robust scatter estimator.
* ``RobustSOBI`` is a package-specific composition of robust whitening, weighted
  lagged scatter, and established SOBI joint diagonalization.
* ``TwoScatterICA`` uses an established two-scatter ICA principle with a
  robustcov-specific bounded radial second scatter and implementation choices.
* Experimental approximations are identified explicitly and are not presented
  as exact reproductions of a paper.
* ``DistributionallyRobustPCA`` evaluates the exact weighted-Wasserstein risk
  for each candidate, but currently optimizes only over a deterministic path;
  it does not claim global Grassmann optimization or RWPI radius calibration.
* Benchmark wins are claims about the documented scenarios, not universal
  state-of-the-art claims.

Language used in the documentation
----------------------------------

Preferred wording includes:

* "our implementation";
* "the robustcov adaptation";
* "a package-specific composition";
* "best among the tested methods in this scenario";
* "independently implemented from the published description".

The project avoids unsupported wording such as "we invented robust ICA",
"universally state of the art", or "the first implementation".

Adding a new estimator
----------------------

A new public estimator must add:

#. a canonical entry in ``robustcov.provenance.METHOD_PROVENANCE``;
#. primary references in ``REFERENCE_CATALOG`` and ``docs/references.bib``;
#. a clear provenance label and implementation-differences note;
#. benchmark, validation, performance-gate, or workflow ownership;
#. tests confirming both provenance and evidence coverage.

The provenance tests intentionally fail when a canonical public estimator is
added without this information.
