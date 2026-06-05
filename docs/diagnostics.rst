Diagnostics
===========

Robust distance diagnostics are central to the package. They help users decide whether a covariance estimate is meaningful for anomaly detection or whether the tail behavior is too far from Gaussian assumptions.

Diagnostic report
-----------------

.. code-block:: python

   report = rc.diagnostic_report(est)
   print(report.summary())

The report includes radial kurtosis, QQ tail deviation, condition number, detected fraction, and recommendations.

Distance profile / proline plot
-------------------------------

.. code-block:: python

   rc.plot_robust_distance_profile(est, output_path="profile.png", show=False)

This plot shows sorted robust distances and the threshold. It makes the tail and threshold crossing easy to inspect.

Distance panel
--------------

.. code-block:: python

   rc.plot_robust_distance_panel(est, output_path="panel.png", show=False)

The panel combines a distance profile, histogram, and chi-square QQ plot.

2D anomaly diagnostics
----------------------

.. code-block:: python

   rc.plot_anomaly_scatter_2d(est, X[:, :2], labels=y, output_path="scatter.png", show=False)

For real high-dimensional data, use PCA or domain features before plotting.
