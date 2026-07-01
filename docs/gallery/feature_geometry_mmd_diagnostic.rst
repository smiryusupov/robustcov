MMD with a robust feature-space metric
======================================

This example treats feature vectors as learned representations produced by an
upstream model.  ``robustcov`` only sees the feature matrix.

A fitted :class:`robustcov.FeatureGeometry` induces an RBF kernel through
Mahalanobis-style distances.  The example then computes ordinary kernel MMD
between an old/reference feature distribution and a new/current distribution
using that geometry-induced kernel.

This is not a new MMD theory contribution.  It is MMD with a kernel whose
feature-space metric is estimated robustly from reference features.

The intended pattern is:

* clean empirical metric MMD detects the shifted distribution;
* contaminated empirical metric MMD can be weakened by leverage contamination;
* robust contaminated metric MMD remains close to clean-reference behavior.

Run the example with:

.. code-block:: bash

   python examples/feature_geometry_mmd_diagnostic.py

Representative output
---------------------

.. literalinclude:: ../_static/gallery/feature_geometry_mmd_diagnostic_output.txt
   :language: text

Interpretation
--------------

When empirical covariance is contaminated in a sensitive low-variance direction,
the induced RBF kernel can make reference and shifted distributions look too
similar.  A robust scatter estimate gives a more stable feature-space metric, so
the resulting MMD remains sensitive to the shift.
