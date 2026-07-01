:orphan:

Scope and non-claims
--------------------

This design note treats robust covariance estimators as geometry constructors
for feature matrices. The intended use case is to fit a stable reference
geometry and reuse it for distances, kernels, class-conditional scores, and
distribution-level drift diagnostics.

The scope is deliberately limited:

* this is a feature-geometry layer, not a replacement ML pipeline;
* it is not a universal OOD detector;
* it is not an adversarial defense;
* it is not a new MMD estimator or new MMD theory;
* MMD-style examples use ordinary kernel MMD with a robust feature-space metric;
* adversarial examples, if mentioned, are exploratory and should be treated as
  future work unless detector-aware adaptive attacks are evaluated.

The strongest practical framing is embedding or feature monitoring under a
possibly contaminated reference window.

Robust feature geometry
=======================

Motivation
----------

Modern machine-learning systems often compute distances, similarities, kernels,
retrieval scores, drift statistics, or out-of-distribution scores in learned
representation spaces. These representations may come from pretrained image
models, text embedding models, autoencoders, penultimate neural-network layers,
or task-specific feature extractors.

Those feature spaces can themselves be contaminated. They may contain outliers,
label noise, corrupted samples, heavy-tailed directions, leverage points, domain
shift, or adversarially perturbed examples. If downstream geometry is built from
ordinary empirical covariance, then Mahalanobis distances, whitening maps,
kernels, MMD-style statistics, and nearest-neighbor similarities may inherit the
same instability.

The goal of this direction is to make ``robustcov`` a lightweight robust
geometry layer for learned representations.

Core pipeline
-------------

The intended workflow is::

   pretrained model / embedding model / autoencoder / feature extractor
   ↓
   feature matrix
   ↓
   robust covariance or scatter estimation
   ↓
   robust precision, whitening, distances, kernels, or drift diagnostics
   ↓
   OOD, anomaly detection, retrieval, similarity, MMD, or monitoring workflow

The package should not train deep neural networks. It should operate on feature
matrices produced elsewhere.

Interpretability angle
----------------------

The feature-geometry layer is useful not only because it can improve a metric,
but because it makes representation-space failures inspectable.

When empirical covariance is fitted on contaminated features, the fitted metric
may inflate variance along leverage directions.  Mahalanobis distances, nearest
class scores, RBF kernels, and drift statistics can then become insensitive
exactly in the directions where sensitivity is needed.

A robust scatter estimate gives a more stable central geometry.  This makes it
possible to compare:

* clean empirical geometry;
* contaminated empirical geometry;
* clean robust geometry;
* contaminated robust geometry.

If the contaminated empirical geometry collapses while the contaminated robust
geometry remains close to the clean-reference behavior, the diagnostic points to
metric pollution rather than to a failure of the downstream OOD or retrieval
task itself.

Scope
-----

This direction should add tools for:

* robust global feature geometry;
* robust class-conditional feature geometry;
* robust Mahalanobis scores on learned features;
* robust whitening of feature vectors;
* robust RBF-style kernels induced by fitted scatter estimates;
* robust similarity and retrieval filtering in embedding spaces;
* MMD-style two-sample diagnostics using robust feature-space geometry;
* covariance and scatter drift monitoring between feature distributions.

Non-goals
---------

This direction should not turn ``robustcov`` into:

* a deep-learning training framework;
* a replacement for PyTorch, scikit-learn, GPyTorch, or dedicated OOD libraries;
* a universal OOD detector;
* a new MMD-style diagnostic with a robust feature-space metric theory package;
* a benchmark leaderboard project;
* a claim that robust covariance solves all deep-learning robustness problems.

The correct claim is narrower:

   ``robustcov`` provides robust scatter geometry tools for contaminated
   learned representation spaces.

Initial API sketch
------------------

A minimal global feature-geometry wrapper could look like:

.. code-block:: python

   geom = rc.FeatureGeometry(
       estimator=rc.RegularizedCauchy(alpha=0.10),
   ).fit(Z_train)

   scores = geom.mahalanobis_scores(Z_test)
   Z_white = geom.transform(Z_test)
   K = geom.rbf_kernel(Z_test, Z_train, length_scale=1.0)

The wrapper should mostly compose existing primitives:

* robust scatter estimators;
* robust precision computation;
* robust Mahalanobis distances;
* whitening;
* existing robust kernel helpers.

A class-conditional version could look like:

.. code-block:: python

   geom = rc.ClassConditionalFeatureGeometry(
       estimator=rc.RegularizedCauchy(alpha=0.10),
   ).fit(Z_train, y_train)

   scores = geom.ood_scores(Z_test)
   labels = geom.predict_nearest_class(Z_test)

This would support Lee-style class-conditional Mahalanobis workflows while
allowing empirical covariance to be replaced by robust scatter estimates.

Dependency policy
-----------------

The core implementation should depend only on the existing ``robustcov`` runtime
stack. It should not require PyTorch, torchvision, transformers, or large data
downloads.

Examples using deep models or external datasets should live in optional examples
or external example scripts. Small synthetic examples and small precomputed
feature arrays are preferred for normal documentation and CI.

Candidate experiments
---------------------

The first research examples should be small and focused.

Synthetic contaminated feature space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate feature vectors with class structure and controlled contamination:

* heavy-tailed features;
* radial outliers;
* angular/leverage outliers;
* label noise;
* small-sample high-dimensional regimes.

Compare empirical Mahalanobis geometry against robust scatter geometry using
AUROC, AUPR, FPR95, condition number, failure rate, and runtime.

Frozen deep-feature OOD
~~~~~~~~~~~~~~~~~~~~~~~

Use frozen features from a pretrained model. The package should consume the
features as arrays rather than training the model itself.

Compare:

* empirical class-conditional Mahalanobis;
* robust class-conditional Mahalanobis;
* global robust Mahalanobis;
* simple softmax or distance baselines when available.

Robust embedding retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use contaminated reference embeddings and evaluate retrieval quality before and
after robust leverage filtering or robust similarity scoring.

Metrics may include:

* top-1 label match;
* precision at k;
* fraction of leverage artifacts appearing in top-k results.

Robust feature-distribution comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare ordinary RBF MMD-style diagnostics against robust geometry versions:

* ordinary Euclidean RBF kernel;
* robust Mahalanobis RBF kernel;
* robustly whitened features;
* contamination-only versus true-drift scenarios.

The goal is not to claim a new MMD theory, but to show that robust feature-space
geometry can improve practical two-sample and drift diagnostics under
contamination.

Milestone plan
--------------

Milestone 1: design and scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add this design note and agree on scope, non-goals, dependency policy, and first
examples.

Milestone 2: global feature geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a small ``FeatureGeometry`` wrapper around existing estimators, distances,
whitening, and kernels.

Milestone 3: class-conditional feature geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add class-wise robust scatter fitting and Mahalanobis-style OOD scoring.

Milestone 4: robust feature drift and MMD-style diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add examples and lightweight helpers for robust feature-distribution comparison.

Milestone 5: paper-quality examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a small set of reproducible examples around synthetic features, frozen
deep features, embedding retrieval, and drift/MMD diagnostics.

Acceptance criteria
-------------------

The first implementation PR should be considered successful if it:

* adds a small wrapper without duplicating estimator logic;
* keeps deep-learning libraries optional;
* works on plain NumPy feature matrices;
* has tests using synthetic data;
* has one gallery example;
* documents limitations clearly;
* avoids claims of universal OOD or deep-learning robustness.