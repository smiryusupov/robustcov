Quickstart
==========

Robust covariance under contamination
-------------------------------------

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

.. code-block:: python

   est = rc.RegularizedCauchy(alpha=0.10).fit(X)
   report = rc.diagnostic_report(est)
   print(report.summary())

Automatic estimator selection
-----------------------------

.. code-block:: python

   auto = rc.AutoRobustScatter(selection="diagnostic").fit(X)
   print(auto.summary())
   cov = auto.covariance_
