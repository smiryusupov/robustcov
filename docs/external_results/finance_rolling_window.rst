Rolling-window finance anomaly detection
========================================

From isolated days to market regimes
------------------------------------

A single unusual day may be noise.  This example instead summarizes 20-day
windows and scores each window, so sustained changes in volatility, correlation,
drawdown, or cross-asset behavior appear as a run of high values.

Rolling-window construction
---------------------------

This documented run uses the same reproducible synthetic price table as the
single-day market-stress example.  The script forms 20-trading-day windows with
a 5-day step, producing 176 windows over 8 assets.

Command
-------

.. code-block:: bash

   python examples_external/finance_rolling_window_anomaly.py \
     --prices examples_external/data/prices.csv \
     --window 20 \
     --step 5 \
     --outdir results/external/finance_rolling_window

Console output
--------------

.. literalinclude:: ../_static/external_results/finance_rolling_window/output.txt
   :language: text

Summary metrics
---------------

.. list-table:: Rolling-window finance result
   :header-rows: 1

   * - Method
     - Windows
     - Window length
     - Step
     - Assets
     - Detected windows
     - Threshold
     - Max distance
     - Radial kurtosis
   * - RegularizedCauchy
     - 176
     - 20
     - 5
     - 8
     - 5
     - 152.37
     - 214.03
     - 3.98

Plots
-----

.. figure:: ../_static/external_results/finance_rolling_window/top_stress_windows.png
   :alt: Rolling-window finance top anomalous windows
   :width: 95%

   Top anomalous rolling windows.  The top three windows overlap the September
   2020 stress period, showing that the method detects regimes rather than only
   isolated points.

.. figure:: ../_static/external_results/finance_rolling_window/rolling_distance_profile.png
   :alt: Rolling-window finance ranked profile
   :width: 95%

   Ranked window-level robust-distance profile.  Windows above the threshold are
   the first regime candidates to inspect.

Overlapping stress windows
--------------------------

The rolling example detects 5 windows above the threshold.  The top windows are:

.. list-table:: Top anomalous windows
   :header-rows: 1

   * - Rank
     - Start date
     - End date
     - Robust distance
   * - 1
     - 2020-09-03
     - 2020-09-30
     - 214.03
   * - 2
     - 2020-08-27
     - 2020-09-23
     - 195.33
   * - 3
     - 2020-09-10
     - 2020-10-07
     - 186.59
   * - 4
     - 2022-09-01
     - 2022-09-28
     - 171.02
   * - 5
     - 2022-08-25
     - 2022-09-21
     - 158.05

The leading windows overlap.  That pattern is expected when the underlying
change lasts for several weeks and is more informative than one isolated high
score.

Estimator choice
----------------

Use ``RegularizedCauchy`` when windows are high-dimensional or heavy-tailed.  If
window features are smoother and closer to elliptical Student-t behavior, try
``StudentTScatter`` as a sensitivity check.

Using the signal in monitoring
------------------------------

For real markets, use rolling-window anomalies as a monitoring layer.  Review
clusters of high-scoring windows, not just one row at a time.  Consider adding
volatility, drawdown, correlation, and sector-return features before fitting the
robust scatter model.
