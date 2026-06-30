Robust feature-geometry similarity kernels
==========================================

This example treats feature vectors as learned representations produced by an
upstream model.  ``robustcov`` only sees the feature matrix.

A fitted :class:`robustcov.FeatureGeometry` induces an RBF similarity kernel
through Mahalanobis-style distances.  The diagnostic asks whether this kernel
remains sensitive to a shift in a low-variance feature direction when the
reference geometry is contaminated by leverage-like points.

The intended pattern is:

* clean empirical geometry separates reference and shifted features;
* contaminated empirical geometry can make shifted features spuriously similar;
* robust contaminated geometry remains close to clean-reference behavior.

This is a mechanism check, not a universal kernel-learning benchmark.

Run the example with:

.. code-block:: bash

   python examples/feature_geometry_similarity_kernel.py

Representative output
---------------------

.. literalinclude:: ../_static/gallery/feature_geometry_similarity_kernel_output.txt
   :language: text

Interpretation
--------------

The RBF kernel is computed from distances induced by a fitted feature geometry.
When empirical covariance is contaminated in a sensitive low-variance direction,
shifted features can remain spuriously similar to reference features.  A robust
scatter estimate gives a more stable feature metric, so kernel similarity to
shifted features drops as expected.
