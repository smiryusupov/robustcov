Robust covariance geometry as a layer
=====================================

``robustcov`` is a lightweight geometry layer for data matrices, feature
matrices, and learned representations.

It is not a replacement for scikit-learn, PyTorch, GPyTorch, kernel libraries,
or drift-detection frameworks. Instead, it provides robust covariance geometry
that can plug into those workflows.

The core chain is:

.. code-block:: text

   data or features
   -> robust scatter / covariance
   -> robust precision matrix
   -> Mahalanobis distance
   -> whitening
   -> robust kernels
   -> GP input metrics
   -> MMD-style drift diagnostics
   -> anomaly, retrieval, OOD, finance, sensors, and monitoring workflows

Why a geometry layer?
---------------------

Many machine-learning workflows depend on a reference geometry:

* anomaly detectors score distance from a reference distribution;
* OOD methods compare test features to training features;
* kernels depend on a distance metric;
* Gaussian processes depend on an input metric;
* drift detectors compare reference and new-batch distributions;
* retrieval systems compare embeddings.

If the reference data or reference features are contaminated, ordinary empirical
covariance can distort this geometry. A few leverage points can inflate
variance in important directions and weaken the resulting distance, kernel, or
drift signal.

``robustcov`` addresses this by fitting robust covariance or scatter estimators
and exposing the induced geometry.

What robustcov provides
-----------------------

The package provides estimators and utilities for:

* robust covariance and scatter estimation;
* robust Mahalanobis distances;
* whitening and precision matrices;
* robust RBF-style kernels;
* robust input metrics for Gaussian-process workflows;
* sklearn-compatible anomaly detectors;
* feature-space geometry through ``FeatureGeometry`` and
  ``ClassConditionalFeatureGeometry``;
* MMD-style distribution diagnostics using a robust feature-space metric.

The MMD component is ordinary kernel MMD. The contribution is the robust metric
used inside the kernel, not a new MMD estimator.

Feature geometry
----------------

A typical learned-representation workflow looks like this:

.. code-block:: python

   import robustcov as rc

   geom = rc.FeatureGeometry(
       estimator=rc.FastMCD(random_state=0),
   ).fit(X_ref)

   scores = geom.decision_function(X_new)
   K = geom.rbf_kernel(X_new, length_scale=1.0)

Here ``X_ref`` may be embeddings from a model, latent vectors from an
autoencoder, tabular features, or hidden-layer representations from a neural
network.

Practical embedding monitoring
------------------------------

A practical use case is monitoring embedding drift when the reference window
may be contaminated.

The gallery example
:doc:`gallery/feature_geometry_embedding_monitoring` demonstrates the workflow:

.. code-block:: text

   reference embeddings
   new-batch embeddings
   contaminated reference window
   central reference-anchor selection
   MMD-style drift calibration
   empirical metric vs robust metric

In that example, empirical geometry keeps contaminated points inside the central
reference anchor and loses most of the drift signal. Robust geometry excludes
the contaminated reference points and preserves the drift signal.

Scope and non-claims
--------------------

``robustcov`` is deliberately scoped.

It is:

* a robust covariance geometry package;
* a tool for constructing distances, kernels, precision matrices, and
  feature-space metrics;
* useful for contaminated reference distributions and monitoring workflows.

It is not:

* a universal OOD detector;
* an adversarial defense;
* a replacement for deep-learning frameworks;
* a replacement for scikit-learn;
* a new MMD estimator;
* a benchmark-driven claim of state-of-the-art performance.

The safest way to think about the package is:

.. code-block:: text

   robustcov provides robust covariance geometry that other ML workflows can use.
