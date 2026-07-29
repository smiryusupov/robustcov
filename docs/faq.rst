Practical FAQ
=============

This page answers the questions that usually arise after the quickstart.  It is
organized around decisions rather than around individual estimators.  For a
full method table, see :doc:`estimator_guide`; for end-to-end patterns, see
:doc:`workflows`.

Choosing a method
-----------------

Where should I start?
~~~~~~~~~~~~~~~~~~~~~

Start from the failure mode and the fitted quantity you need:

* **Separable row outliers, with** :math:`n` **comfortably larger than**
  :math:`p`: start with ``FastMCD``.
* **Diffuse heavy tails or a small sample:** compare ``RegularizedCauchy`` and
  ``StudentTScatter``.
* **Row contamination with** :math:`p` **close to or larger than** :math:`n`:
  use ``MRCD`` or a regularized heavy-tail estimator.
* **Isolated bad cells or missing entries:** use ``CellMCD``, ``CellRCov``, or
  ``CellPCA`` depending on whether you need covariance or a low-rank model.
* **One matrix that is low rank plus sparse gross corruption:** use
  ``PrincipalComponentPursuit``.
* **A fixed reference subspace that must be monitored:** use
  ``RobustSubspaceMonitor``.
* **A subspace that is allowed to evolve gradually:** consider the experimental
  ``OnlineRobustSubspaceTracker``.

The :doc:`method_comparison` page explains which methods estimate comparable
quantities and which comparisons are scientifically inappropriate.

What is the difference between covariance, scatter, and shape?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A covariance matrix has an identified scale.  A scatter matrix describes
multivariate spread but may use a robust scale convention that differs from the
ordinary sample covariance.  A shape matrix, such as Tyler shape, is identified
only up to a positive scalar.

This distinction matters when you compare numerical values, build likelihoods,
or use absolute Mahalanobis thresholds.  Shape estimators are often excellent
for directions, whitening, and relative geometry, but an application that needs
physical covariance units must add an explicit scale estimate.

How do rowwise, cellwise, and heavy-tail contamination differ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rowwise contamination** means that an entire observation may be unreliable.
**Cellwise contamination** means that a few entries can be bad while the rest of
the row remains useful.  **Heavy tails** mean that large observations may be
legitimate draws from the regular population rather than a separate outlier
mechanism.

These settings require different methods.  A high-breakdown rowwise estimator
can discard too much information under isolated bad cells, while a heavy-tail
estimator may not isolate a compact adversarial cluster.  The contamination
model is part of the scientific assumption, not merely a tuning choice.

What should I use when the number of features is close to or exceeds the sample size?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid relying on an unregularized covariance inverse.  Typical starting points
are:

* ``MRCD`` for rowwise contamination with target regularization;
* ``RegularizedCauchy`` or ``RegularizedTyler`` for heavy-tailed or elliptical
  data;
* ``CellRCov`` when bad cells and missingness coexist with genuine low-rank
  structure;
* ``SGLASSO`` or ``RobustGraphicalLasso`` when the final goal is a sparse
  precision graph.

``FastMCD`` is not the default when a nonsingular clean subset cannot exist.

Can RobustCov choose the estimator automatically?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AutoRobustScatter`` and ``AutoRobustAnomalyDetector`` compare a bounded set of
candidates using package diagnostics.  They are useful for exploration and
sensitivity analysis, but they are not oracles.  Inspect the selected estimator,
its fitted diagnostics, and the stability of the choice before using it in a
production decision.

PCA, decomposition, and latent structure
----------------------------------------

Which robust PCA method should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The phrase *robust PCA* refers to several different problems:

* ``RobustPCA``: robust covariance/scatter PCA for rowwise outliers or heavy
  tails, with score- and orthogonal-distance diagnostics.
* ``PrincipalComponentPursuit`` / ``PCP``: decompose one matrix as low rank plus
  sparse, potentially very large entrywise corruption.
* ``CellPCA``: fit a low-rank model with bad cells, abnormal rows, and missing
  entries.
* ``SparseCellPCA``: add sparse, interpretable loadings to the CellPCA setting.
* ``DensityPowerRobustPCA``: directly fit a complete low-rank model with a
  tunable density-power loss.
* ``DistributionallyRobustPCA``: study a principal subspace under a specified
  covariance-shift ambiguity geometry.
* ``RobustMultilinearPCA``: preserve matrix or tensor modes instead of flattening
  structured observations.

These estimators do not solve the same optimization problem.  Choose from the
contamination mechanism, data structure, and output you need rather than from
the shared PCA label.

Does Principal Component Pursuit handle new observations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PrincipalComponentPursuit.fit`` decomposes the matrix supplied during fitting.
Its ``transform`` method projects new rows onto the learned low-rank row space;
it does not solve a fresh sparse-corruption decomposition for every new row.
Use PCP when the matrix-level decomposition itself is the target.  Use a robust
PCA estimator or a monitor when repeated out-of-sample scoring is central.

What should I use for ICA, temporally correlated sources, or factor models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``TwoScatterICA`` for independent components identified through two robust
scatter matrices, ``RobustSOBI`` when temporal lag structure identifies the
sources, and ``RobustFactorModel`` for a static low-rank factor representation.
Recovery metrics account for permutation, sign, and scale indeterminacy; raw
component ordering is not meaningful by itself.

Monitoring and anomaly alerts
-----------------------------

How should I choose an anomaly threshold?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit the geometry on training/reference data, reserve a separate calibration
sample, and convert scores into alerts with ``ConformalAlertCalibrator`` when
its exchangeability assumption is defensible.  The conformal layer provides
finite-sample p-values and makes the score-to-alert step explicit.

Do not calibrate and evaluate on the same observations.  For dependent time
series, regime changes, or contaminated calibration samples, ordinary split
conformal guarantees may not apply; use block-aware validation and report the
assumptions.

What is the difference between a frozen monitor and an adaptive tracker?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RobustSubspaceMonitor`` keeps the fitted reference subspace fixed and measures
departure from that baseline.  It is appropriate when the original regime must
remain the definition of normal.

The experimental ``OnlineRobustSubspaceTracker`` can update a subspace slowly
through bounded robust mini-batches.  It is appropriate when gradual evolution
is expected, but it can absorb persistent changes into the new reference.  Use
its update diagnostics and change safeguards rather than treating adaptation as
inherently desirable.

Can I combine online tracking with conformal calibration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, but decide which reference the alert should represent.  A common pattern
is to score a batch and compute conformal p-values **before** updating the
tracker.  Keep the calibrator frozen when alerts must remain relative to the
original regime.  Recalibrate only through an explicit reviewed procedure when
the operational definition of normal changes.

Does the adversarial spectral filter replace MCD or heavy-tail estimators?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No.  ``SpectralFilteringCovariance`` is an experimental estimator for a bounded
fraction of adversarial whole-row replacements under approximately Gaussian or
sub-Gaussian regular data.  It is not the default for diffuse heavy tails,
cellwise errors, or an unknown contamination mechanism.  See
:doc:`adversarial_covariance_filtering` for its assumptions and the guarantees
that do not carry over from the theoretical filtering literature.

Data structure and missingness
------------------------------

Which methods accept missing values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing-value support is method-specific.  ``CellMCD``, ``CellRCov``,
``CellPCA``, and the cellwise/multilinear workflows are designed to retain useful
information around missing or corrupted cells.  Many classical covariance,
precision, PCP, and source-separation estimators require complete finite arrays.
Check the estimator page rather than assuming that all sklearn-style objects
share the same missing-data behavior.

What should I use for matrix- or tensor-valued observations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``MMCD`` when each observation is a matrix and a separable row/column
covariance model is scientifically meaningful.  Use ``RobustMultilinearPCA``
when the target is a low-rank multilinear representation with mode-specific
loadings.  Flattening is still possible, but it discards structure and can make
the covariance problem much larger.

Can I use RobustCov with learned embeddings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes.  Robust covariance, feature geometry, PCA, conformal calibration, and
subspace monitoring operate on numerical feature vectors regardless of whether
they came from tabular measurements or a neural encoder.  RobustCov does not
train the encoder; it supplies a geometry and monitoring layer around the
resulting vectors.  See :doc:`feature_geometry` and the embedding examples in
:doc:`use_case_gallery`.

API, performance, and reproducibility
-------------------------------------

Are the estimators compatible with scikit-learn workflows?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Public estimators follow sklearn-style constructor, ``fit``, fitted-attribute,
and parameter-introspection conventions.  Core interoperability is tested with
``clone``, ``Pipeline``, and ``GridSearchCV``.  Not every object is a drop-in
replacement for a scikit-learn covariance estimator because some methods return
shape matrices, decompositions, structured outputs, or monitoring records.

What do stable, provisional, and experimental mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installed ``robustcov/_public_api.json`` manifest classifies every exported
name.  Stable interfaces have the strongest compatibility commitment.
Provisional interfaces are supported but may evolve before 1.0.  Experimental
interfaces can change substantially and must document where their
implementation differs from the cited research.  See :doc:`api_stability`.

Is the compiled extension required?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No.  RobustCov has Python fallbacks and supports a native-free wheel.  The
optional C++/OpenMP extension accelerates selected kernels but does not define a
separate API.  Use ``robustcov.native_available()`` and ``robustcov.has_openmp()``
to inspect the installed build.

Are results deterministic with OpenMP enabled?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set every estimator's ``random_state`` where available and control thread counts
for repeatable benchmarks.  Parallel floating-point reductions can still differ
at the last few bits because summation order changes.  Such tiny differences are
normal; large changes in selected supports, ranks, or alerts should be
investigated.

Does RobustCov use a GPU?
~~~~~~~~~~~~~~~~~~~~~~~~~

The core package targets NumPy/SciPy arrays on CPU, with optional native OpenMP
acceleration.  It does not require CUDA, train neural networks, or keep tensors
on a GPU.  Convert learned representations to finite two-dimensional arrays
before fitting the relevant geometry estimator.

How should I read the galleries and benchmark claims?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`use_case_gallery` shows how methods fit application workflows.  The
:doc:`benchmark_gallery` reports scenario-specific evidence, controls, and
failure modes.  A benchmark result applies only to the tested quantity,
contamination model, dimensions, tuning rules, and implementation versions.  It
is not a universal ranking of estimators.

Will importing RobustCov download external data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No.  External downloads are always explicit.  Dataset loaders use a
user-controlled cache, verify available checksums or fingerprints, and support
offline ``download=False`` behavior.  Read the Docs displays reviewed aggregate
snapshots; it does not download or run external datasets during the normal
Sphinx build.  See :doc:`external_data`.
