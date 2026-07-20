Conformal alert calibration
============================

``ConformalAlertCalibrator`` converts any scalar anomaly, residual-risk, or
monitoring score into a conservative conformal p-value. It separates two
questions that are often mixed together:

* the robust estimator defines **what counts as unusual geometry**;
* the conformal calibrator defines **how extreme a new score is relative to a
  held-out reference sample**.

Basic workflow
--------------

Split acceptable reference data into a model-fitting set and a calibration set.
For anomaly scores where larger values are more unusual:

.. code-block:: python

   import robustcov as rc

   detector = rc.RobustOutlierDetector(
       estimator=rc.RegularizedCauchy(alpha=0.10),
       threshold="empirical",
   ).fit(X_train)

   # RobustOutlierDetector.score_samples returns larger values for normal data,
   # so negate it to obtain an upper-tail anomaly score.
   calibration_scores = -detector.score_samples(X_calibration)

   calibrator = rc.ConformalAlertCalibrator(
       alpha=0.05,
       tail="upper",
   ).fit(calibration_scores)

   new_scores = -detector.score_samples(X_new)
   p_values = calibrator.p_values(new_scores)
   alerts = calibrator.predict_alerts(new_scores)
   labels = calibrator.predict(new_scores)  # 1 normal, -1 alert

The score model must not be fitted on ``X_calibration``. Reusing calibration
observations to train or tune the score model can invalidate the ordinary split
conformal interpretation.

Finite-sample resolution
------------------------

With ``n`` calibration scores, the smallest attainable conservative p-value is
``1 / (n + 1)``. At ``alpha=0.05``, at least 19 calibration scores are needed for
any alert to be possible. Inspect:

.. code-block:: python

   print(calibrator.min_p_value_)
   print(calibrator.resolution_limited_)
   print(calibrator.calibration_summary())

Ties are handled conservatively. ``threshold_`` is an interpretable strict
score boundary, while ``predict_alerts`` always uses the conformal p-values and
is the preferred decision method.

Monitoring scores
-----------------

The calibrator accepts scores from any frozen monitoring design. For example,
calibrate a scalar window statistic on stable windows and evaluate future
windows:

.. code-block:: python

   calibrator = rc.ConformalAlertCalibrator(alpha=0.01).fit(
       stable_window_risks
   )
   p_value = calibrator.p_values(current_window_risk)
   if calibrator.predict_alerts(current_window_risk):
       print(f"alert: conformal p={p_value:.4g}")

The C-MAPSS external protocol uses this pattern for independently calibrated
window reconstruction risk.

Assumptions and limits
----------------------

The usual finite-sample marginal interpretation requires exchangeability
between calibration scores and future inlier scores. Ordinary split conformal
calibration does **not** automatically provide:

* conditional guarantees for every operating regime or subgroup;
* validity under arbitrary temporal dependence or adaptive repeated testing;
* correction for covariate or distribution shift;
* online false-discovery or false-alarm-rate control;
* protection against adversarial contamination of the calibration set.

Large upper-tail contaminants in the reference set often make upper-tail
p-values conservative, at the cost of power. RobustCov exposes that basic
calibration behavior but does not implement the active-cleaning procedure of
Bashari, Sesia, and Romano (2025).

API
---

See :class:`robustcov.ConformalAlertCalibrator` and the validation script
``benchmarks/conformal_alert_calibration_validation.py``.
