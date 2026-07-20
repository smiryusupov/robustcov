Workflows
=========

Start from the statistical task rather than from an estimator name. Most
RobustCov workflows follow the same pattern: fit a robust geometry on a
reference sample, turn that geometry into task-specific scores or structure,
and validate the result under the contamination and shift model that matters.

Covariance, scatter, and anomaly scoring
----------------------------------------

Use robust covariance or scatter when empirical covariance is distorted by
rowwise contamination, heavy tails, high dimension, bad cells, or structured
observations.

Typical path:

#. choose an estimator with :doc:`estimator_guide`;
#. fit it on a reference or training sample;
#. compute robust Mahalanobis distances or use
   :class:`~robustcov.RobustOutlierDetector`;
#. calibrate operational alerts on held-out scores with
   :class:`~robustcov.ConformalAlertCalibrator` when exchangeability is a
   defensible approximation;
#. inspect distance, support, and conditioning diagnostics.

See :doc:`quickstart`, :doc:`conformal_alert_calibration`, and
:doc:`gallery_methods/robust_estimators`.

PCA, subspaces, and monitoring
------------------------------

Use robust PCA when the goal is low-rank representation, reconstruction,
row/cell diagnostics, or monitoring changes relative to a stable reference.

``RobustPCA`` supplies score and orthogonal distances, ``CellPCA`` handles mixed
casewise/cellwise contamination and missing values, and
``RobustSubspaceMonitor`` compares rolling windows with a frozen reference
model. ``SubspaceStability`` adds bootstrap diagnostics for loadings and
principal angles.

See :doc:`monitoring`, :doc:`conformal_alert_calibration`, and
:doc:`gallery_methods/pca_factor_models`.

Features, embeddings, and kernels
---------------------------------

``FeatureGeometry`` converts learned representations into robust distances,
whitened features, and similarity kernels. Use it for one-class screening,
retrieval filtering, class-conditional geometry, or drift summaries without
requiring RobustCov to train the upstream representation model.

See :doc:`feature_geometry` and
:doc:`gallery_topics/biomedical_images_embeddings`.

Sparse precision and conditional structure
------------------------------------------

Use ``RobustGraphicalLasso`` when a robust covariance estimate should be turned
into a sparse precision graph. Use ``SGLASSO`` for radial heavy-tail regimes
where spatial-sign geometry is appropriate. Interpret graph recovery only under
a benchmark or scientific design that matches the assumed data-generating
structure.

See :doc:`sparse_precision` and :doc:`spatial_sign_precision`.

Structured and cellwise data
----------------------------

Use ``CellMCD``, ``CellRCov``, ``CellPCA``, or ``SparseCellPCA`` when isolated
cells can be corrupted or missing. Use ``MMCD`` for matrix-valued observations
with separable row/column covariance, and ``RobustMultilinearPCA`` for robust
low-rank tensor structure.

See :doc:`cellwise_covariance`, :doc:`cellwise_regularized_covariance`,
:doc:`matrix_covariance`, and :doc:`robust_multilinear_pca`.

Latent sources and factor models
--------------------------------

Use ``TwoScatterICA`` for independent components, ``RobustSOBI`` for temporally
correlated sources, and ``RobustFactorModel`` for static common-factor
structure. These methods rely on distinct identifiability assumptions; compare
source or subspace recovery with permutation-aware metrics.

See :doc:`source_separation_factor_models` and
:doc:`gallery_methods/ica_source_separation`.
