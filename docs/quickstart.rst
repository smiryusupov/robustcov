Quickstart
==========

Robust covariance under contamination
-------------------------------------

This example creates a dataset with a contaminated tail and fits ``FastMCD``.
When outliers pull empirical covariance away from the central data cloud,
``FastMCD`` estimates a more stable covariance structure and gives robust
Mahalanobis distances for anomaly screening.

.. code-block:: python

   import numpy as np
   import robustcov as rc

   rng = np.random.default_rng(0)
   X = rng.normal(size=(500, 5))
   X[:40] += 8.0

   est = rc.FastMCD(quality="balanced", random_state=0).fit(X)
   print(est.location_)
   print(est.radial_kurtosis_)

   rc.plot_robust_distance_panel(est, output_path="distance_panel.png", show=False)

Small-sample heavy-tail scatter
-------------------------------

Heavy-tailed data can produce unstable empirical covariance estimates,
especially when the sample size is not large. Regularized robust scatter
estimators provide a more stable geometry while still allowing heavy-tailed
variation.

.. code-block:: python

   est = rc.RegularizedCauchy(alpha=0.10).fit(X)
   report = rc.diagnostic_report(est)
   print(report.summary())

Automatic estimator selection
-----------------------------

When you are not sure which robust estimator to use, ``AutoRobustScatter`` gives
a practical starting point. It fits a candidate estimator and exposes the same
``location_``, ``covariance_``, and ``precision_`` attributes used by the rest of
the package.

.. code-block:: python

   auto = rc.AutoRobustScatter(selection="diagnostic").fit(X)
   print(auto.summary())
   cov = auto.covariance_


Where to go next
----------------

After the quickstart, see :doc:`estimator_guide` for estimator selection,
:doc:`use_case_gallery` for application examples, :doc:`benchmark_gallery` for
evidence and comparisons, and :doc:`geometry` for robust SPD geometry utilities.

