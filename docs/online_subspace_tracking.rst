Online robust subspace tracking
===============================

``OnlineRobustSubspaceTracker`` follows a principal subspace that is expected to
change gradually while isolated cells or entire observations may be corrupted.
It is intended for sensor streams, evolving embeddings, and other micro-batch
workflows where a permanently frozen reference would eventually become stale.

The estimator is experimental. It is inspired by projected-residual robust
subspace tracking and online outlier-robust PCA, but it is **not NORST** and does
not carry NORST's sparse-support recovery, tracking-delay, or memory-optimality
guarantees.

How it updates
--------------

For every incoming batch the tracker:

#. projects observations onto the current subspace;
#. standardizes the orthogonal residual feature by feature;
#. replaces a small number of extreme residual cells by the current
   reconstruction;
#. rejects rows that remain jointly extreme in score and orthogonal geometry;
#. appends accepted cleaned observations to a bounded recent-sample buffer;
#. periodically fits a robust PCA candidate on that buffer; and
#. interpolates the current and candidate projectors when the candidate obeys
   the configured slow-change safeguard.

This design separates adaptation from alerting. Use
:class:`~robustcov.ConformalAlertCalibrator` on a held-out scalar score when a
calibrated alert layer is needed.

Basic use
---------

.. code-block:: python

   import robustcov as rc

   tracker = rc.OnlineRobustSubspaceTracker(
       n_components=3,
       estimator=rc.RegularizedCauchy(alpha=0.10),
       update_interval=64,
       buffer_size=256,
       adaptation_rate=0.5,
       max_update_angle=30.0,
   ).fit(X_initial)

   result = tracker.update(X_microbatch)

   print(result.n_accepted)
   print(result.n_rejected)
   print(result.n_cell_corrections)
   print(result.change_detected)
   print(tracker.components_)

``partial_fit`` performs the same update and returns the estimator for
incremental-pipeline compatibility.

Frozen monitoring versus adaptive tracking
-------------------------------------------

Use :class:`~robustcov.RobustSubspaceMonitor` when normality must remain tied to
a fixed reference period. Use ``OnlineRobustSubspaceTracker`` when the normal
subspace itself is expected to evolve slowly. A common production design uses
both:

* a frozen monitor for policy or safety boundaries;
* an adaptive tracker for descriptive state estimation;
* a reviewed rule deciding whether and when the adaptive state may replace a
  production baseline.

Robustness model
----------------

The implementation targets two practical corruption patterns:

``isolated projected-residual cells``
   A small number of feature values are inconsistent with the current
   low-dimensional reconstruction. They are replaced before candidate fitting.

``dense row outliers``
   An observation remains extreme both outside and within the current subspace
   after limited cell repair. It is scored but excluded from adaptation.

The method can fail when a dense outlier cloud forms a coherent alternative
subspace, when change is too abrupt for the recent buffer, or when corrupted
coordinates are not identifiable from projected residuals.

Important parameters
--------------------

``update_interval``
   Accepted observations required before a candidate update.

``buffer_size``
   Maximum number of recent cleaned observations used for robust candidate
   fitting. Larger values stabilize updates but increase delay.

``adaptation_rate``
   Projector interpolation weight. A value of one accepts the candidate
   subspace directly; smaller values smooth the trajectory.

``cell_threshold`` and ``max_cell_fraction``
   Control limited cell repair. They are engineering thresholds, not estimated
   contamination probabilities.

``change_detection_angle`` and ``max_update_angle``
   Report and constrain candidate/current principal-angle changes. The latter is
   a slow-change safeguard, not a hypothesis test.

Diagnostics
-----------

Each update returns :class:`~robustcov.OnlineSubspaceUpdate` with accepted and
rejected counts, repaired-cell counts, per-row anomaly scores, masks, candidate
angle, update status, and subspace version. Aggregate counters remain available
on the fitted estimator.

Research relationship
---------------------

NORST uses projected compressive sensing to estimate sparse outliers and then
updates the subspace on cleaned mini-batches. Its guarantees require conditions
including slow subspace change, incoherence, sparse corruption, and lower bounds
on most outlier magnitudes. Residual-based online robust PCA instead studies
one-pass robust sampling with recovery guarantees comparable to batch methods.

``OnlineRobustSubspaceTracker`` adopts the broad projected-residual and online
update ideas but deliberately uses a simpler robustcov composition. Cite the
package and the primary research, and do not describe this class as NORST.

See also
--------

* :doc:`monitoring`
* :doc:`conformal_alert_calibration`
* :doc:`robust_pca`
* :doc:`gallery/online_robust_subspace_tracking`
