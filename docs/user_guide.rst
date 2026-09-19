User guide
==========

Start with the task, not with an estimator name.  RobustCov contains many
methods because different contamination models require different geometry; you
do not need to learn the full catalog before using the package.

If you are new to RobustCov, use this sequence:

#. run the :doc:`quickstart` to learn the fitted-object pattern;
#. use :doc:`estimator_guide` to identify the failure mode in your data;
#. follow one task family below;
#. use the :doc:`api` only when you need exact constructor parameters or fitted
   attributes.

Choose the task
---------------

.. list-table:: Task-first map
   :header-rows: 1
   :widths: 26 35 39

   * - I need to...
     - Start with
     - Read next
   * - estimate covariance or scatter when some rows are abnormal
     - ``FastMCD``, ``DetS``, ``DetMM``, or ``MRCD``
     - :doc:`estimator_guide` and :doc:`workflows`
   * - estimate stable geometry under heavy tails or high dimension
     - ``RegularizedCauchy``, ``StudentTScatter``, ``RegularizedTyler``, or ``MRCD``
     - :doc:`estimator_guide` and :doc:`method_comparison`
   * - preserve useful rows when individual cells are bad or missing
     - ``CellMCD``, ``CellRCov``, ``CellPCA``, or ``SparseCellPCA``
     - :doc:`cellwise_covariance` and :doc:`cellwise_pca`
   * - reduce dimension or separate low-rank structure from corruption
     - ``RobustScatterPCA``, ``PrincipalComponentPursuit``, ``CellPCA``, or ``RobustMultilinearPCA``
     - :doc:`robust_pca`, :doc:`principal_component_pursuit`, and :doc:`cellwise_pca`
   * - score unusual observations or calibrate alerts
     - ``RobustOutlierDetector`` and ``ConformalAlertCalibrator``
     - :doc:`workflows` and :doc:`conformal_alert_calibration`
   * - monitor a reference subspace or feature distribution over time
     - ``RobustSubspaceMonitor`` or ``FeatureGeometry``
     - :doc:`monitoring` and :doc:`feature_geometry`
   * - estimate a sparse precision or dependence graph
     - ``RobustGraphicalLasso`` or ``SGLASSO``
     - :doc:`sparse_precision` and :doc:`spatial_sign_precision`
   * - work with matrix- or tensor-valued observations
     - ``MatrixMCD`` or ``RobustMultilinearPCA``
     - :doc:`matrix_covariance` and :doc:`robust_multilinear_pca`
   * - recover latent sources or factors
     - ``TwoScatterICA``, ``RobustSOBI``, or ``RobustFactorModel``
     - :doc:`source_separation_factor_models`

Covariance, scatter, and robust distances
-----------------------------------------

Most RobustCov workflows begin by estimating a geometry on a reference sample.
Use rowwise high-breakdown estimators when a minority of complete observations
are contaminated, and regularized or heavy-tail estimators when the main issue
is conditioning or diffuse tails.

After fitting, common outputs are ``location_``, ``covariance_``,
``precision_``, support/weight diagnostics, and robust Mahalanobis distances.
The :doc:`robust_geometry_layer` explains how these fitted quantities connect
across the package.

Bad cells and structured observations
--------------------------------------

A rowwise estimator can discard too much information when only a few entries in
each row are corrupted.  Use the cellwise family when clean cells in otherwise
imperfect rows should still contribute.  Matrix- and tensor-valued estimators
are separate again: they assume scientifically meaningful structure across
modes rather than a generic flat feature vector.

See :doc:`cellwise_covariance`, :doc:`cellwise_regularized_covariance`,
:doc:`matrix_covariance`, and :doc:`robust_multilinear_pca`.

PCA, decomposition, and latent structure
----------------------------------------

There are several superficially similar low-rank tasks:

* ``RobustScatterPCA`` performs ordinary eigendecomposition on a robust scatter
  estimate;
* ``PrincipalComponentPursuit`` separates one matrix into low-rank and sparse
  components;
* ``CellPCA`` and ``SparseCellPCA`` model low-rank structure with cellwise and
  casewise robustness;
* source-separation and factor-model classes target different identifiability
  assumptions from PCA.

Start with :doc:`workflows` if you are unsure which of those problems you have.
The detailed mathematical descriptions live in the :doc:`algorithms`.

Detection, calibration, and monitoring
--------------------------------------

An anomaly score and an operational alert are not the same object.  First fit a
reference geometry and produce scores.  Then calibrate or validate the alerting
rule on held-out data whose exchangeability or shift assumptions you can state.
For changing systems, distinguish a frozen-reference monitor from an adaptive
tracker.

See :doc:`conformal_alert_calibration`, :doc:`monitoring`, and
:doc:`online_subspace_tracking`.

Precision, geometry, and integrations
-------------------------------------

Sparse precision estimation, SPD distances, kernels, explanation references,
and learned-feature geometry are downstream uses of robust scatter.  They are
useful once the base statistical task is clear; they are not prerequisites for
learning the package.

Use :doc:`api` for exact signatures, :doc:`use_case_gallery` for runnable
examples, and :doc:`benchmark_gallery` for validation evidence.
