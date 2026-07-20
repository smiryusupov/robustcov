Quickstart
==========

Robust covariance and outlier detection
---------------------------------------

This example creates heavy-tailed data with a contaminated subset, fits
``FastMCD``, and flags the largest 7.5% of robust distances.  Pass an unfitted
estimator to ``RobustOutlierDetector``; the detector clones and fits it without
mutating the original object.

.. literalinclude:: _snippets/quickstart_outlier_detection.py
   :language: python
   :linenos:

The fitted scatter estimator is available as ``det.estimator_`` and can be used
with the diagnostic plotting helpers:

.. code-block:: python

   rc.plot_robust_distance_panel(
       det.estimator_, output_path="distance_panel.png", show=False
   )

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

Rolling production monitoring
-----------------------------

Use a frozen robust reference when batches arrive over time and the type of
drift matters operationally.

.. code-block:: python

   monitor = rc.RobustSubspaceMonitor(
       n_components=0.95,
       estimator=rc.RegularizedCauchy(alpha=0.10),
       window_size=100,
       alarm_patience=2,
   ).fit(X_reference)

   result = monitor.update(X_new_batch)
   if result.ready:
       print(result.summary())
       print(result.exceeded)

See :doc:`monitoring` for calibration and interpretation details.

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
evidence and comparisons, and :doc:`geometry` for robust SPD geometry utilities, and :doc:`monitoring` for frozen-reference rolling drift diagnostics.

