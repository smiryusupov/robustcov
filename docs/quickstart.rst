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

Calibrate anomaly alerts on held-out scores
---------------------------------------------

A robust distance is a ranking score, not automatically a calibrated alert.
Fit the score model on training data and calibrate a separate reference split:

.. code-block:: python

   detector = rc.RobustOutlierDetector(
       estimator=rc.RegularizedCauchy(alpha=0.10),
       threshold="empirical",
   ).fit(X_train)

   calibrator = rc.ConformalAlertCalibrator(alpha=0.05).fit(
       -detector.score_samples(X_calibration)
   )
   alerts = calibrator.predict_alerts(-detector.score_samples(X_new))

See :doc:`conformal_alert_calibration` for the exchangeability assumption,
finite-sample p-value resolution, and monitoring use.

Low-rank plus sparse decomposition
----------------------------------

When one matrix is the sum of a low-rank signal and sparse, arbitrarily large
cell corruption, use Principal Component Pursuit rather than a scatter-based
PCA estimator:

.. code-block:: python

   pcp = rc.PrincipalComponentPursuit(tol=1e-7).fit(X)
   low_rank = pcp.low_rank_
   sparse_corruption = pcp.sparse_
   print(pcp.decomposition_summary())

See :doc:`principal_component_pursuit` for the incoherence/sparsity assumptions
and the distinction from :class:`robustcov.RobustPCA`.

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



Track a slowly changing subspace
--------------------------------

When a fixed baseline is too restrictive, initialize an experimental online
tracker on an acceptable reference block and update it with micro-batches:

.. code-block:: python

   tracker = rc.OnlineRobustSubspaceTracker(
       n_components=4,
       update_interval=64,
       buffer_size=256,
       adaptation_rate=0.5,
   ).fit(X_initial)

   result = tracker.update(X_next)
   print(result.n_accepted, result.change_detected)

The tracker repairs isolated projected-residual cells, rejects dense row
outliers, and adapts only through bounded robust mini-batch updates. See
:doc:`online_subspace_tracking` for assumptions and limitations.
