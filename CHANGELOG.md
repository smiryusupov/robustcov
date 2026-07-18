# Changelog

## Unreleased
- Added deterministic multivariate `DetS` and `DetMM` estimators with Tukey-bisquare S-scale and fixed-scale MM refinement.
- Added experimental ``SparseCellPCA`` for cellwise/casewise robust low-rank fitting with exact-zero elastic-net loadings.
- Added ``KMRCD`` for robust subset-based anomaly detection in linear, RBF, polynomial, callable, or precomputed kernel spaces.

Highlights:
- Added estimator-driven `RobustPCA` with robust projection, reconstruction, whitening, and variance-based component selection.
- Added score-distance and orthogonal-distance diagnostics plus a robust PCA outlier-map plot.
- Added robust PCA tests, documentation, and a synthetic example.
- Added production embedding monitoring, yield-curve factor, and cross-asset market-risk RobustPCA gallery examples.
- Added `MRCD`, a minimum regularized covariance determinant estimator for rowwise contamination in high-dimensional data.
- Added MRCD mathematics, tests, API documentation, and a high-dimensional outlier example.
- Added `MMCD` for robust matrix-valued location and Kronecker covariance estimation, including signed cell/row/column distance contributions.
- Added MMCD mathematics, tests, plotting support, and a multichannel sensor-window example.
- Added `CellMCD` for cellwise contamination and missing values, with conditional residual diagnostics, corrected-data transforms, tests, and a market-data example.
- Added `CellPCA` for simultaneous cellwise errors, abnormal rows, and missing entries, with weighted low-rank diagnostics and a process-spectra example.
- Added experimental `CellRCov` for high-dimensional full covariance estimation under mixed cellwise/casewise contamination and missingness, including residual-shrinkage selection and a benchmarked example.
- Added `RobustGraphicalLasso` for sparse precision matrices from robust scatter estimates, including ADMM optimization, EBIC selection, graph diagnostics, and a contaminated market-network example.
- Added `SubspaceStability` for bootstrap loading, eigenvalue, explained-variance, and principal-angle diagnostics around robust PCA fits.
- Extended `SubspaceStability` with moving-block, circular-block, stationary, and cluster bootstrap designs for serially dependent and grouped observations.


## 0.0.2

Release-readiness and repository metadata update.

Highlights:
- Added JOSS paper draft and bibliography.
- Added verified benchmark-claim tracker.
- Added citation, contributing, and changelog metadata.
- Added release-readiness audit checklist.
- Fixed Windows/headless CI plotting tests by using a non-interactive Matplotlib backend.
- No public API changes.

## 0.0.1

Initial public-development release.

Highlights:

- FastMCD / MinCovDet-style robust covariance estimator.
- Tyler and regularized Tyler shape estimators.
- Student-t and Cauchy-style regularized robust scatter estimators.
- Robust anomaly diagnostics and plotting utilities.
- Auto robust scatter selection and cluster-aware multimodal diagnostics.
- Robust kernel/input metric utilities for sklearn-style and GPyTorch-style workflows.
- SPD geometry utilities for robust scatter diagnostics and covariance comparison.
- Optional OpenMP acceleration for compiled C++ paths.
- Sphinx documentation, use-case gallery, benchmark gallery, and optional external/Kaggle examples.