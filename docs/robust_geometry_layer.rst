What RobustCov does
===================

``robustcov`` is a robust multivariate-geometry library. It estimates location,
scale, covariance, scatter, precision, or low-rank structure when ordinary
empirical covariance is too sensitive or too unstable.

The fitted geometry can then be reused for:

* Mahalanobis anomaly scores and diagnostic plots;
* whitening and full-matrix distance metrics;
* principal components and reconstruction diagnostics;
* rolling subspace and feature-distribution monitoring;
* sparse precision and conditional-dependence estimation;
* matrix-valued covariance, cellwise-robust models, and multilinear PCA;
* robust ICA, SOBI, and latent-factor estimation.

The package is useful when data contain a minority of abnormal rows, isolated
bad cells, broad heavy tails, leverage points, missing entries, a difficult
``p``-to-``n`` ratio, or a train-to-deployment covariance shift that can be
stated explicitly.

The geometry layer
------------------

Many downstream algorithms depend on a covariance matrix even when covariance
estimation is not the final task. Mahalanobis distance, whitening, RBF kernels,
Gaussian-process input metrics, PCA, and several monitoring statistics all need
a notion of scale and direction.

``robustcov`` supplies that notion from a robust fit:

.. code-block:: text

   observations, windows, or learned features
       -> robust location and scatter or robust low-rank fit
       -> covariance, precision, whitening, or principal subspace
       -> scores, kernels, graphs, diagnostics, or monitoring

A vector-data example
---------------------

.. code-block:: python

   import robustcov as rc

   estimator = rc.RegularizedCauchy(alpha=0.10).fit(X_reference)

   scores = estimator.mahalanobis(X_new)
   covariance = estimator.covariance_
   precision = estimator.precision_

The common fitted attributes make estimators composable with the rest of the
package. The correct estimator still depends on the contamination model; see
:doc:`estimator_guide` rather than treating the catalog as a universal ranking.

A feature-geometry example
--------------------------

.. code-block:: python

   geom = rc.FeatureGeometry(
       estimator=rc.FastMCD(random_state=0),
   ).fit(X_reference)

   distances = geom.mahalanobis_scores(X_new)
   K = geom.rbf_kernel(X_new, length_scale=1.0)

``X_reference`` may contain text embeddings, image features, sensor summaries,
or latent vectors from an autoencoder. The encoder remains outside
``robustcov``.

A subspace-monitoring example
-----------------------------

.. code-block:: python

   monitor = rc.RobustSubspaceMonitor(
       n_components=0.95,
       estimator=rc.RegularizedCauchy(alpha=0.10),
       window_size=256,
   ).fit(X_reference)

   result = monitor.update(X_batch)
   if result.ready:
       print(result.summary())
       print(result.exceeded)

The reference model stays fixed while later batches are scored. See
:doc:`monitoring` for calibration and interpretation.

What the package does not do
----------------------------

``robustcov`` does not train neural networks, provide a complete production
monitoring service, infer causal graphs, or automatically determine whether the
scientific contamination model is rowwise, cellwise, heavy-tailed, multimodal,
or distributional. It provides numerical estimators, diagnostics, and reusable
geometry that can be composed with scikit-learn, PyTorch, GPyTorch, and domain
workflows.

Start with :doc:`quickstart` for a runnable example, :doc:`workflows` for a
task-oriented map, and :doc:`benchmark_gallery` for evidence and limitations.
