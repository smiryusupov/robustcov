:orphan:

Robust feature-geometry drift detection
=======================================

This example treats feature vectors as learned representations produced by an
upstream model.  ``robustcov`` only sees the feature matrix.

The diagnostic compares an old/reference feature distribution with a
new/current feature distribution.  The reference set is deliberately polluted by
leverage-like contamination in the same low-variance direction where the new
distribution later shifts.

The intended pattern is:

* clean empirical geometry detects the drift;
* contaminated empirical geometry can be blinded by leverage contamination;
* robust contaminated geometry remains close to clean-reference behavior.

This is a mechanism check, not a universal drift-detection benchmark.

Run the example with:

.. code-block:: bash

   python examples/feature_geometry_drift_detection.py

Representative output
---------------------

.. literalinclude:: ../_static/gallery/feature_geometry_drift_detection_output.txt
   :language: text

Interpretation
--------------

The example illustrates a representation-space failure mode: empirical
covariance can inflate a contaminated low-variance direction and make a later
shift in that direction look ordinary.  A robust scatter estimate gives a more
stable central reference geometry, so the same drift remains visible under
Mahalanobis-style feature scores.
