Robust geometry for feature vectors
===================================

``FeatureGeometry`` fits a location and scatter model to a matrix of feature
vectors, then uses that fit for distances, whitening, and kernels.  The feature
vectors may come from an image encoder, a text embedding model, an autoencoder,
a tabular pipeline, or any other model that produces a numeric representation.

``robustcov`` does not train the representation model.  It starts from the
feature matrix produced by that model.

Global feature geometry
-----------------------

.. code-block:: python

   import robustcov as rc

   geom = rc.FeatureGeometry(
       estimator=rc.FastMCD(n_init=40, random_state=0),
   ).fit(Z_train)

   distances = geom.mahalanobis_scores(Z_test)
   Z_white = geom.transform(Z_test)
   K = geom.rbf_kernel(Z_test, Z_train, length_scale=1.0)

The fitted scatter matrix defines which directions count as large or small.
Using a robust estimator limits the influence of contaminated reference vectors
on that geometry.

The class provides:

* Mahalanobis distances from the fitted center;
* whitening and inverse transformation;
* pairwise squared distances under the fitted precision matrix;
* RBF kernels based on those distances.

Inputs are ordinary two-dimensional arrays with shape
``(n_samples, n_features)``.

Class-conditional geometry
--------------------------

``ClassConditionalFeatureGeometry`` fits one geometry per label.  It can return
the nearest class, all class-wise distances, or the distance to the closest
class model.

.. code-block:: python

   geom = rc.ClassConditionalFeatureGeometry(
       estimator=rc.FastMCD(n_init=40, random_state=0),
   ).fit(Z_train, y_train)

   nearest_class = geom.predict(Z_test)
   ood_scores = geom.decision_function(Z_test)
   class_distances = geom.class_mahalanobis_scores(Z_test)

This is a distance-based classifier and diagnostic.  It does not include class
priors or covariance log-determinants, so it should not be interpreted as a
Gaussian discriminant model.

Score direction
---------------

``decision_function`` follows a distance convention: larger values are farther
from the global fit or from every fitted class geometry.  This differs from
some scikit-learn anomaly estimators, where larger decision values indicate
more typical observations.

Embedding drift example
-----------------------

The :doc:`embedding monitoring example
<gallery/feature_geometry_embedding_monitoring>` fits empirical and robust
geometries to the same contaminated reference window.  It then selects a
central reference subset and compares new batches with an RBF-kernel MMD
statistic.

In that simulation, the empirical covariance includes several contaminated
vectors in the reference anchor.  FastMCD excludes them, leaving a cleaner
anchor and a stronger signal when the new batch shifts.

Related examples
----------------

* :doc:`Synthetic global OOD scoring
  <gallery/feature_geometry_synthetic_ood>`
* :doc:`Class-conditional OOD scoring
  <gallery/feature_geometry_class_conditional_ood>`
* :doc:`Production embedding monitoring with RobustPCA
  <gallery/robust_pca_embedding_monitoring>`

Scope
-----

These classes provide covariance-based geometry for features produced
elsewhere.  They do not replace an embedding model, an OOD benchmark suite, a
retrieval system, or a full drift-detection platform.  Their role is narrower:
fit a robust metric and make it available to the rest of the application.
