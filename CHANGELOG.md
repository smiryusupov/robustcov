# Changelog

## Unreleased

### Fixed

- Hardened native array boundaries with checked dimensions and allocation products, explicit finite-value validation, and an opt-in AddressSanitizer/UndefinedBehaviorSanitizer workflow.
- Replaced FastMCD's Wilson--Hilferty calibration approximations with exact SciPy chi-square cutoffs and Gaussian consistency factors for the raw and reweighted covariance estimates.
- Made FastMCD reject ``initial_c_steps=0`` instead of silently running one C-step, and corrected ``converged_``, ``n_iter_``, and objective diagnostics to describe the selected fitted solution.

## 0.1.0 - 2026-07-29

### Highlights

- First normal `0.1` release of RobustCov.
- Added MRCD and kernel MRCD for high-dimensional and nonlinear robust geometry.
- Added robust PCA, structured-data, cellwise-contamination, monitoring, calibration, sparse-precision, and latent-structure workflows.
- Added platform wheels for CPython 3.12–3.14 on Linux, macOS, and Windows.
- Added task-oriented documentation, examples, benchmarks, method provenance, and reviewed external validation.
- Hardened release automation, installed-wheel smoke testing, Read the Docs builds, and Trusted Publishing.

## 0.1.0a3 - 2026-07-29

### Changed

- Include the logo URL correction
- wheel-test cleanup
- external workflow fix
- test cleanup.

## Unreleased

### Changed

- Refined the documentation branding with a clean icon-only covariance mark, a centered vertical hero lockup, and balanced sidebar logo alignment.
- Added the RobustCov visual identity to the documentation sidebar, favicon, README, and a reorganized product-focused landing page with three clear entry paths.
- Replaced the estimator-era FAQ with a task-oriented practical FAQ covering contamination models, high-dimensional choices, PCA and decomposition variants, monitoring and conformal calibration, structured data, API maturity, and deployment behavior.

## 0.1.0a1 - 2026-07-20

### Release engineering

- Added exact version/tag checks across package metadata, runtime exports,
  citation metadata, and the public API manifest.
- Added a manual TestPyPI Trusted Publishing rehearsal with post-publication
  installation tests outside the source checkout.
- Added deterministic SHA-256 release-artifact manifests and strengthened the
  C-MAPSS snapshot schema with summary validation, candidate diagnostics, and
  explicit interpretation of overlapping PCA/DRO-PCA results.

### Changed

- Simplified public documentation by removing the defensive project-claims page; method provenance and evidence rules now live in contributor guidance and automated checks.
- Added a machine-readable public API stability manifest and release checks that require every top-level and experimental export to be classified.
- Split the pytest suite into explicit unit, integration, statistical, benchmark, native, and packaging groups for predictable CI execution.
- Standardized source-tree subprocess environments so benchmark and example tests no longer require manually exporting `PYTHONPATH`.

- Added ``PrincipalComponentPursuit`` / ``PCP`` for canonical low-rank-plus-sparse matrix decomposition, with inexact-ALM optimization, cell/row/column diagnostics, convergence history, validation, provenance, API documentation, and an example.

- Added experimental ``SpectralFilteringCovariance`` for bounded adversarial whole-row contamination, with matrix-free quadratic filtering, diagnostics, validation, provenance, API documentation, and an example.

- Added experimental ``OnlineRobustSubspaceTracker`` for bounded-memory adaptation to gradually changing principal subspaces, with projected-residual cell repair, dense-row rejection, slow-change safeguards, diagnostics, benchmarks, and explicit non-NORST provenance.
- Added ``ConformalAlertCalibrator`` for conservative split-conformal p-values and alert labels from arbitrary anomaly or monitoring scores, including finite-sample resolution diagnostics and C-MAPSS integration.
- Added conformal-calibration validation, provenance, API documentation, and a task-oriented workflows page required by the documentation landing page.
- Removed the optional Sphinx viewcode extension to keep strict Read the Docs builds within predictable resource limits; repository source links remain available through the Furo theme.
- Simplified the example-gallery navigation so the sidebar exposes only method
  and application-domain landing pages; individual examples remain linked from
  those pages without expanding into a long global list.
- Reorganized the documentation and README around user goals, clarified the package
  promise and boundaries, grouped method pages hierarchically, and focused the
  public external benchmark gallery on reviewed C-MAPSS snapshots.
- Add a reviewed C-MAPSS FD002/FD004 snapshot publisher, provenance manifest,
  Read the Docs validation, and a staged roadmap for public-dataset evidence.
- Retain UCI Gas Sensor Drift as an explicit-cache loader and exploratory local
  analysis, outside the reviewed public snapshot gallery.

- Added cache-only external dataset loaders for UCI Gas Sensor Array Drift and NASA C-MAPSS, including atomic downloads, checksum/fingerprint validation, and safe nested-ZIP extraction.
- Added real-data DRO-PCA protocols for temporal gas-sensor drift and operating-regime-aware C-MAPSS degradation monitoring without committing raw data.
- Added offline fake-archive tests, external-data documentation, ignore rules, and a manually triggered external-data GitHub Actions workflow.
- Added experimental ``DistributionallyRobustPCA`` with an exact weighted-Wasserstein scalar-dual risk, anisotropic transport geometries, identity-geometry PCA control, and a deterministic candidate-path optimizer.
- Added held-out distribution-shift benchmarks, gallery plots, provenance metadata, and tests separating DRO-PCA from contamination-robust PCA.
- Added a runtime method-provenance registry covering every canonical public estimator and core source-separation algorithms.
- Added generated method/reference documentation, a machine-readable BibTeX catalog, and an explicit project-contributions/claims policy.
- Added tests that prevent new public estimators from shipping without attribution, implementation notes, and benchmark ownership.
- Expanded `CITATION.cff` to instruct users to cite both the software release and the methodological papers they use.
- Expanded the task-specific benchmark suite to cover robust ICA, classical and robust SOBI, robust PCA, and robust factor models with permutation/sign/scale-aware recovery metrics.
- Added a benchmark coverage inventory mapping canonical public estimators to comparative benchmarks, validation gates, performance gates, or end-to-end workflows.
- Added latent-structure benchmark plots, Sphinx pages, Monte Carlo integration, and one-command report integration.
- Reorganized runnable examples by ICA/source separation, PCA/factor models, robust estimators, and anomaly-monitoring workflows, with separate ICA, SOBI, and factor-model gallery pages.
- Added ``TwoScatterICA`` with robust scatter whitening and bounded radial second-scatter contributions.
- Added classical ``SOBI`` and ``RobustSOBI`` with robust lagged scatter estimation.
- Added ``RobustFactorModel`` with spatial-Kendall factor selection and Huber alternating fits.
- Added permutation/sign/scale-aware BSS metrics and a shared symmetric joint diagonalizer.
- Added a C++ Jacobi joint-diagonalization kernel retained by the complete-SOBI 1.5x acceleration gate.
- Added release-readiness validation for source metadata, wheels, and source distributions.
- Added minimum-dependency CI for Python 3.12, 3.13, and 3.14 and version-specific NumPy/SciPy/scikit-learn lower bounds.
- Migrated package license metadata to the PEP 639 SPDX format and declared `LICENSE` and `NOTICE` explicitly.
- Documented the public deprecation policy and expanded the reproducible release checklist.
- Fixed CI so test dependencies are installed explicitly and the test suite can run without Matplotlib.
- Hardened sparse precision, spatial-sign precision, feature geometry, and subspace monitoring under extreme rescaling, feature permutations, singular/repeated features, and constant inputs.
- Added experimental `SpatialSignGraphicalLasso` / `SGLASSO` for sparse shape-precision graphs under heavy-tailed elliptical data.
- Added experimental `DensityPowerRobustPCA`, a direct Gaussian density-power-divergence low-rank estimator with cell weights and robust PCA diagnostics.
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