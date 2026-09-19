Quickstart
==========

This page gives one complete first workflow: fit a robust reference geometry,
score observations, and inspect the fitted result.  You do not need to choose
among the full method catalog to get started.

Fit and score observations
--------------------------

The example below creates heavy-tailed data with a contaminated subset and fits
one robust outlier detector.  ``RobustOutlierDetector`` clones and fits the
scatter estimator supplied to it, so the fitted estimator is available as
``det.estimator_``.

.. literalinclude:: _snippets/quickstart_outlier_detection.py
   :language: python
   :linenos:

The important fitted outputs are:

* ``det.labels_``: inlier/outlier labels for the fitted sample;
* ``det.estimator_.location_`` and ``det.estimator_.covariance_``: the fitted
  robust geometry;
* ``det.mahalanobis(X)``: robust squared distances for observations in the same
  feature space.

For a compact diagnostic view:

.. code-block:: python

   rc.plot_robust_distance_panel(
       det.estimator_, output_path="distance_panel.png", show=False
   )

Calibrate alerts on held-out scores
-----------------------------------

A robust distance is a ranking score, not automatically a calibrated alert. If
an application needs alert probabilities or a controlled significance level,
fit the score model on training data and calibrate a separate reference split:

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

Where to go next
----------------

.. list-table:: Continue from the task you have
   :header-rows: 1
   :widths: 42 58

   * - If you need to...
     - Go to
   * - decide which robust method matches your contamination model
     - :doc:`estimator_guide`
   * - understand covariance, PCA, cellwise, monitoring, or structured-data workflows
     - :doc:`user_guide`
   * - follow an end-to-end task recipe
     - :doc:`workflows`
   * - see runnable examples by application domain
     - :doc:`use_case_gallery`
   * - inspect benchmark evidence and limitations
     - :doc:`benchmark_gallery`
   * - look up exact parameters, methods, and fitted attributes
     - :doc:`api`
