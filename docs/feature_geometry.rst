Robust feature geometry
=======================

``robustcov`` can be used as a robust geometry layer on top of learned
representations.

The package does not train representation models.  Instead, it operates on
feature matrices produced by other systems: image encoders, text embedding
models, autoencoders, penultimate neural-network layers, tabular feature
pipelines, or other embedding methods.

Motivation
----------

Many modern machine-learning workflows compute distances, similarities,
retrieval scores, kernels, drift statistics, or out-of-distribution scores in
feature space.  If the reference feature set is contaminated, heavy-tailed, or
contains leverage points, then ordinary empirical covariance can distort that
geometry.

``FeatureGeometry`` wraps an existing robust scatter estimator and exposes:

* robust Mahalanobis scores;
* robust whitening;
* pairwise squared distances in the fitted metric;
* RBF-style kernels induced by the fitted robust feature metric.

Basic usage
-----------

.. code-block:: python

   import robustcov as rc

   geom = rc.FeatureGeometry(
       estimator=rc.FastMCD(n_init=40, random_state=0),
   ).fit(Z_train)

   scores = geom.mahalanobis_scores(Z_test)
   anomaly_scores = geom.decision_function(Z_test)
   Z_white = geom.transform(Z_test)
   K = geom.rbf_kernel(Z_test, Z_train, length_scale=1.0)

The input arrays are ordinary NumPy-style feature matrices with shape
``(n_samples, n_features)``.

Class-conditional geometry
--------------------------

For labeled feature matrices, ``ClassConditionalFeatureGeometry`` fits one
feature geometry per class and exposes nearest-class and OOD-style scores.

.. code-block:: python

   geom = rc.ClassConditionalFeatureGeometry(
       estimator=rc.FastMCD(n_init=40, random_state=0),
   ).fit(Z_train, y_train)

   nearest_class = geom.predict(Z_test)
   ood_scores = geom.decision_function(Z_test)
   class_distances = geom.class_mahalanobis_scores(Z_test)

This supports class-conditional Mahalanobis workflows on learned features while
keeping the representation model outside ``robustcov``.

Score convention
----------------

``decision_function`` returns distance-style scores: larger values mean farther
from the fitted global geometry or farther from all fitted class geometries.
The API intentionally does not expose ``score_samples`` yet, because that name
often follows the opposite convention in scikit-learn-style anomaly APIs.

Scope
-----

This API is intentionally small.  It does not replace deep-learning libraries,
OOD-detection packages, Gaussian-process frameworks, or retrieval systems.
Instead, it supplies robust covariance geometry for feature representations
created elsewhere.

Good starting examples
----------------------

See :doc:`gallery/feature_geometry_synthetic_ood` for a synthetic example where
empirical feature covariance is distorted by leverage-like contamination, while
robust feature geometry restores separation.

See :doc:`gallery/feature_geometry_class_conditional_ood` for a labeled
feature-space example where robust class-conditional geometry improves
nearest-class OOD scoring under class-wise leverage contamination.
