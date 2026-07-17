Using robust covariance as a geometry
=====================================

Many algorithms rely on a covariance matrix even when covariance estimation is
not the main task.  Mahalanobis distances, whitening, RBF kernels, Gaussian
process input metrics, PCA, and several drift statistics all depend on a notion
of scale and direction.

``robustcov`` supplies that geometry from a robust location and scatter fit.  It
can be used on raw tabular data or on features produced by another model.

From scatter to downstream tools
--------------------------------

A fitted scatter matrix can be reused in several ways:

.. code-block:: text

   observations or features
       -> robust location and scatter
       -> precision matrix and whitening
       -> distances, kernels, PCA, and monitoring

The package includes:

* robust covariance and scatter estimators;
* Mahalanobis distances and anomaly diagnostics;
* whitening and precision-matrix helpers;
* RBF kernels with a full robust metric;
* adapters for scikit-learn and GPyTorch kernels;
* ``FeatureGeometry`` for unlabeled or class-conditional embeddings;
* ``RobustPCA`` for robust subspace estimation;
* ``RobustSubspaceMonitor`` for comparison with a fixed reference period.

Feature vectors
---------------

A feature workflow can start with any two-dimensional array:

.. code-block:: python

   import robustcov as rc

   geom = rc.FeatureGeometry(
       estimator=rc.FastMCD(random_state=0),
   ).fit(X_reference)

   distances = geom.mahalanobis_scores(X_new)
   K = geom.rbf_kernel(X_new, length_scale=1.0)

``X_reference`` might contain text embeddings, image features, sensor summaries,
or latent vectors from an autoencoder.  The encoder remains outside
``robustcov``; only its output is passed to the geometry model.

PCA and rolling comparison
--------------------------

When a low-dimensional representation is useful, ``RobustPCA`` computes
components from the same robust scatter estimates:

.. code-block:: python

   pca = rc.RobustPCA(
       n_components=0.95,
       estimator=rc.RegularizedCauchy(alpha=0.10),
   ).fit(X_reference)

   scores = pca.transform(X_new)
   orthogonal_distance = pca.orthogonal_distances(X_new)

For sequential batches, ``RobustSubspaceMonitor`` keeps the reference fit fixed
and compares it with a robust model fitted to the current window:

.. code-block:: python

   monitor = rc.RobustSubspaceMonitor(
       n_components=0.95,
       estimator=rc.RegularizedCauchy(alpha=0.10),
       window_size=256,
   ).fit(X_reference)

   result = monitor.update(X_batch)
   print(result.exceeded)

See :doc:`robust_pca`, :doc:`monitoring`, and
:doc:`gallery/robust_subspace_monitoring` for the full workflows.

Kernel distribution comparisons
-------------------------------

The MMD helpers in the package use ordinary kernel MMD.  The robust part is the
metric inside the kernel: distances are computed with a robust precision matrix
rather than with unscaled Euclidean distance.  The package does not introduce a
new MMD estimator.

Boundaries of the package
-------------------------

``robustcov`` focuses on estimation and geometry.  It does not train neural
networks, implement a full OOD system, or replace scikit-learn, PyTorch, or a
production monitoring platform.  Its outputs are meant to be composed with
those tools.
