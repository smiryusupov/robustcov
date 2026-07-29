Finance market-stress anomaly detection
=======================================

Daily cross-asset stress scores
-------------------------------

The script assigns one robust Mahalanobis distance to each trading day.  A high
score means that the joint return vector is unusual relative to the central
cross-asset regime; it may reflect market stress, a regime change, or a data
problem.

Price table and injected stress periods
---------------------------------------

The documented run uses a synthetic price table with 899 return days and eight
assets.  Stress periods are injected by the generator, which keeps the example
reproducible and removes any dependency on an external market-data service.

The input format is a CSV with one date column and one numeric price column per
asset:

.. code-block:: text

   date,SPY,QQQ,IWM,TLT,GLD,EFA,EEM,HYG
   2020-01-01,100.0,...

Command
-------

.. code-block:: bash

   python examples_external/finance_market_stress.py \
     --prices examples_external/data/prices.csv \
     --outdir results/external/finance_market_stress

Console output
--------------

.. literalinclude:: ../_static/external_results/finance_market_stress/output.txt
   :language: text

Summary metrics
---------------

.. list-table:: Finance market-stress result
   :header-rows: 1

   * - Method
     - Days
     - Assets
     - Alpha
     - Detected days
     - Threshold
     - Max distance
     - Median distance
     - Radial kurtosis
     - Condition number
   * - RegularizedCauchy
     - 899
     - 8
     - 0.975
     - 23
     - 95.80
     - 251.56
     - 7.34
     - 2.82
     - 6.07

Plots
-----

.. figure:: ../_static/external_results/finance_market_stress/top_stress_days.png
   :alt: Finance market stress top days
   :width: 95%

   Top ranked robust-distance days.  The dashed line is the detection threshold.
   The largest detected day is 2020-09-11 with robust distance 251.56.

.. figure:: ../_static/external_results/finance_market_stress/robust_distance_profile.png
   :alt: Finance market stress ranked profile
   :width: 95%

   Ranked stress-day profile for the top detected dates.  A finance user can
   inspect the dates directly and compare them against known market events or
   injected stress periods.

Detected days
-------------

The estimator flags 23 of 899 days, about 2.6%, which is consistent with the
``alpha=0.975`` threshold.  The condition number is low (about 6.1), so the
robust covariance estimate is numerically stable.  The radial kurtosis is not
extreme, meaning the robust fit has absorbed the central heavy-tailed behavior
without becoming ill-conditioned.

The highest-scoring dates cluster around the injected stress periods,
particularly September 2020 and September 2022.  The output is therefore a
ranked list of dates that can be checked against events, data corrections, or
other risk measures.

Estimator choice
----------------

Start with ``RegularizedCauchy`` for finance returns because it combines strong
radial downweighting with shrinkage.  Use ``StudentTScatter`` as a smoother
heavy-tail sensitivity check, and ``AutoRobustScatter`` if the data regime is
unclear.

Using real market data
----------------------

For real data, run the same script on ETF or stock prices.  Review top days
against market calendars, corporate actions, missing prices, and known stress
events.  For portfolio use, robust distances should complement risk models; they
should not be treated as trading signals without validation.
