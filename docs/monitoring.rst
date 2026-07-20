Monitoring changes in a robust PCA subspace
============================================

``RobustSubspaceMonitor`` compares incoming batches with a fixed reference
period.  It uses a rolling robust PCA fit for the current window, but every new
observation is scored against the original reference before the rolling model
is updated.

This setup is useful when the baseline must remain stable.  A long-running data
problem cannot gradually teach the monitor that the problem is normal.

A first monitor
---------------

Fit the monitor on a period that represents acceptable operation.  The
reference sample must contain more rows than ``window_size``.

.. code-block:: python

   import robustcov as rc

   monitor = rc.RobustSubspaceMonitor(
       n_components=0.95,
       estimator=rc.RegularizedCauchy(alpha=0.10),
       window_size=256,
       calibration_windows=16,
       threshold_quantile=0.99,
       threshold_scale=1.2,
       alarm_patience=2,
   ).fit(X_reference)

   result = monitor.update(X_production_batch)

   if result.ready:
       print(result.summary())
       print(result.exceeded)

The monitor needs a full rolling window before it can compare aggregate
structure.  During warm-up it still returns score and orthogonal distances for
the rows in the incoming batch.

Signals reported by the monitor
-------------------------------

The result keeps several kinds of change separate:

* ``location_shift``: movement of the center in the reference Mahalanobis
  geometry;
* ``scale_shift``: a broad increase or decrease in dispersion;
* ``shape_shift``: a change in covariance structure after removing global
  scale;
* ``max_subspace_angle``: rotation of the leading principal subspace;
* ``score_distance_shift``: observations moving farther along directions that
  were already present in the reference;
* ``orthogonal_distance_shift``: variation appearing outside the retained
  reference subspace;
* outlier fractions for both the complete rolling window and the latest batch.

These quantities point to different follow-up checks.  For example, a location
shift in an embedding service may reflect a population change, while a sharp
orthogonal shift may suggest a preprocessing failure or a new class of input.

Reference model and rolling model
---------------------------------

``reference_model_`` is fitted once and remains unchanged.  ``current_model_``
is refitted on the rolling window after a batch is accepted by
:meth:`~robustcov.RobustSubspaceMonitor.update`.

Use :meth:`~robustcov.RobustSubspaceMonitor.evaluate` when you want to inspect a
batch without changing the rolling state.  ``partial_fit`` is an alias for the
incremental workflow and stores the latest result in ``last_result_``.

Conformal calibration of a scalar window score
-----------------------------------------------

``RobustSubspaceMonitor`` calibrates several decomposed metrics internally. If
your deployment uses one pre-defined scalar risk statistic, use a held-out set
of stable-window scores with :class:`~robustcov.ConformalAlertCalibrator`:

.. code-block:: python

   calibrator = rc.ConformalAlertCalibrator(alpha=0.01).fit(
       stable_window_risks
   )
   current_p = calibrator.p_values(current_window_risk)
   current_alert = calibrator.predict_alerts(current_window_risk)

This gives the score threshold a finite-sample marginal interpretation under
exchangeability. It does not solve temporal dependence, adaptive repeated
testing, or regime-specific calibration. See
:doc:`conformal_alert_calibration` for assumptions and resolution limits.

Calibration and alarms
----------------------

During ``fit``, the monitor draws contiguous windows from the reference period,
fits the same rolling model used in production, and records the resulting drift
metrics.  Aggregate thresholds are empirical quantiles of those reference
values.  Row-level score and orthogonal cutoffs are calibrated separately.

``threshold_scale`` widens or narrows all aggregate thresholds.  Values above
one are often helpful when only a small number of calibration windows is
available.  ``alarm_patience`` controls how many consecutive raw exceedances are
required before ``alarm`` becomes true.

The thresholds are operational cutoffs, not distribution-free hypothesis
tests.  Check the false-alarm rate on a stable period that was not used for
fitting, especially for autocorrelated time series.

Inspecting individual rows
--------------------------

Each update returns distances and flags for the rows in the supplied batch:

.. code-block:: python

   result = monitor.update(X_batch)

   suspicious_rows = X_batch[result.sample_outlier_mask]
   score_distance = result.score_distances
   orthogonal_distance = result.orthogonal_distances

The rolling-window fractions summarize the current monitoring horizon.  The
``batch_*_outlier_fraction`` attributes refer only to the latest batch and react
more quickly to a short burst of unusual records.

Monitoring history
------------------

.. code-block:: python

   rc.plot_subspace_monitor_history(
       monitor,
       output_path="subspace_monitor.png",
       show=False,
   )

The upper panel plots selected metrics after division by their calibrated
thresholds; one is the alert boundary.  The lower panel shows outlier fractions
for the rolling window and the newest batch.

Choosing settings
-----------------

``window_size`` should match the period over which a change needs to persist
before it affects an operational decision.  Smaller windows react faster but
produce noisier covariance and subspace estimates.

Use a regularized estimator when the feature dimension is close to the window
size.  Keep fewer than all components when you need orthogonal-distance or
subspace-rotation diagnostics; with every direction retained, there is no
orthogonal complement to monitor.

Review ``result.exceeded`` even when ``alarm`` is false.  It identifies the
metric that crossed its boundary and is often more informative than the final
combined status.

Computational cost
------------------

The current implementation refits a robust model on each rolling window.  It is
not a constant-memory stochastic PCA algorithm.  Runtime therefore depends on
the scatter estimator, feature dimension, window size, and update frequency.
For high-rate streams, update on micro-batches or run the structural comparison
at a lower cadence.

See also
--------

* :doc:`robust_pca`
* :doc:`feature_geometry`
* :doc:`gallery/robust_subspace_monitoring`
