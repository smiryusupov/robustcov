Kernel MRCD
===========

``KMRCD`` is intended for outlier detection when the regular observations do
not form one approximately elliptical cloud.  Typical examples include curved
spectral trajectories, nonlinear embeddings, and data supplied directly as a
positive-semidefinite similarity matrix.

The method maps observations to a kernel feature space and performs a
regularized high-breakdown subset search there.  Only kernel values are needed;
the feature coordinates are never constructed explicitly.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   model = rc.KMRCD(
       kernel="rbf",
       gamma="median",
       contamination=0.15,
       random_state=0,
   ).fit(X_train)

   distances = model.mahalanobis(X_test)
   labels = model.predict(X_test)

``mahalanobis`` returns squared distances in the kernel feature space.  Larger
values mean that an observation is less compatible with the selected robust
subset.  ``predict`` uses a robust cutoff fitted to the log-transformed training
distances and returns ``1`` for inliers and ``-1`` for outliers.

Kernel and bandwidth
--------------------

The built-in kernels are ``"linear"``, ``"rbf"``, and ``"polynomial"``.  A
callable kernel must accept two two-dimensional arrays and return their Gram
matrix.

For the RBF kernel,

.. math::

   k(x,y)=\exp(-\gamma\|x-y\|^2).

With ``gamma="median"``, the package uses

.. math::

   \gamma = \frac{1}{2\,\operatorname{median}_{i<j}\|x_i-x_j\|^2}

on robustly standardized training observations.  This follows the bandwidth
heuristic used in the KMRCD paper, but it is not a universal optimum.  Inspect
results over a scientifically plausible bandwidth range when the conclusions
matter.

Precomputed kernels
-------------------

KMRCD can be fitted from a square positive-semidefinite kernel matrix:

.. code-block:: python

   model = rc.KMRCD(
       kernel="precomputed",
       contamination=0.15,
   ).fit(K_train)

For new observations, pass their cross-kernel matrix against the training set
and their self-kernel values:

.. code-block:: python

   distances = model.mahalanobis(
       K_test_train,
       kernel_diag=K_test_diag,
   )

The cross-kernel matrix must have shape ``(n_test, n_train)``.  This interface
supports kernels on strings, graphs, or other objects that have no ordinary
numeric feature representation.

What is fitted
--------------

Useful attributes include:

``support_``
   Boolean mask of the selected :math:`h`-subset.

``regularization_``
   Feature-space identity weight :math:`\rho`.

``distances_``
   Squared robust kernel distances of the training observations.

``distance_threshold_``
   Fitted cutoff in squared-distance units.

``regularized_kernel_``
   The regularized centered kernel matrix of the selected subset.

``objective_path_``
   Regularized log-determinant values during the final C-step sequence.

Scope and limitations
---------------------

KMRCD treats complete observations as the unit of contamination.  Use
``CellMCD`` or ``CellRCov`` when isolated cells are corrupted.

The result depends on the kernel.  A nonlinear kernel can reveal useful curved
structure, but it can also create an artificial geometry.  Compare it with a
linear robust baseline and report the bandwidth used.

The package implements the published subset objective and kernel C-steps, but
uses package-specific initial subsets rather than the four refined starts from
the reference implementation.  It should therefore be regarded as an
experimental implementation until broader cross-software validation is
available.

See also
--------

* :doc:`gallery/kmrcd_nonlinear_manifold`
* :doc:`method_comparison`
* :doc:`algorithms`
