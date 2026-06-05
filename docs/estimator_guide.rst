Estimator guide
===============

Which estimator should I use?
-----------------------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Recommended estimator
     - Reason
   * - ``n`` much larger than ``p`` and outliers are separable
     - ``FastMCD``
     - High-breakdown robust covariance with support diagnostics.
   * - Small sample, heavy tails, ``p`` close to or larger than ``n``
     - ``RegularizedCauchy``
     - Strong radial downweighting and shrinkage.
   * - Diffuse heavy tails rather than point anomalies
     - ``StudentTScatter``
     - Smooth heavy-tail M-estimator.
   * - Shape estimation for elliptical data
     - ``RegularizedTyler``
     - Scale-free robust shape estimate.
   * - Unsure which heavy-tail estimator to choose
     - ``AutoRobustScatter``
     - Fits candidates and selects with diagnostic or stability score.

Estimator status
----------------

Stable prototype APIs:

* ``FastMCD``
* ``RegularizedCauchy``
* ``StudentTScatter``
* ``RobustOutlierDetector``
* robust distance plotting helpers

Experimental APIs:

* ``HellingerRegularizedTyler``
* exact KL/Wiesel variants beyond their current alias/prototype behavior
* automatic model selection scores
