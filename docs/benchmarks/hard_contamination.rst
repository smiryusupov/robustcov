Hard contamination scenarios
============================

Question
--------

Where do robust covariance estimators work, and where should users be cautious?

Design
------

This benchmark creates several contamination mechanisms:

* mean-shift outliers;
* clustered contamination;
* variance contamination;
* leverage contamination;
* heavy-tailed inliers.

These scenarios are intentionally not all favorable to MCD.  The goal is to teach users when robust
covariance assumptions match the data and when the geometry is ambiguous.

Results table
-------------

.. csv-table:: Hard contamination scenarios
   :file: ../_static/benchmarks/hard_scenarios.csv
   :header-rows: 1

Interpretation
--------------

Mean-shift and variance contamination are favorable settings for robust covariance.  Clustered and
leverage scenarios can be genuinely ambiguous: the algorithm may not be able to distinguish a bad
cluster from a legitimate subpopulation without domain knowledge.  This is why robust-distance plots
and diagnostic reports are part of the package rather than optional decoration.

Run it yourself
---------------

.. code-block:: bash

   python benchmarks/hard_contamination_scenarios.py --csv results/hard_scenarios.csv

