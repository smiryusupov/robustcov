RobustCov: robust multivariate geometry
========================================

``robustcov`` helps you estimate and use multivariate geometry when empirical
covariance is unreliable. It is designed for contaminated, heavy-tailed,
high-dimensional, incomplete, structured, or shifting data.

The package turns robust location and scatter estimates into practical tools for
covariance recovery, Mahalanobis anomaly scoring, PCA and subspace monitoring,
whitening and kernels, sparse precision estimation, and structured matrix or
tensor analysis. The estimators use sklearn-style ``fit`` methods and expose
familiar fitted attributes such as ``location_``, ``covariance_``, and
``precision_`` where those quantities are identified.

What can I do with it?
----------------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="quickstart.html">
       <div class="gallery-card-placeholder">Score<br>anomalies</div>
       <h3>Detect unusual observations</h3>
       <p>Fit robust covariance, compute Mahalanobis scores, calibrate thresholds, and inspect distance diagnostics.</p>
     </a>
     <a class="gallery-card" href="estimator_guide.html">
       <div class="gallery-card-placeholder">Estimate<br>geometry</div>
       <h3>Recover covariance or scatter</h3>
       <p>Choose methods for rowwise outliers, diffuse heavy tails, bad cells, high dimensions, or matrix-valued observations.</p>
     </a>
     <a class="gallery-card" href="workflows.html#pca-subspaces-and-monitoring">
       <div class="gallery-card-placeholder">PCA<br>monitoring</div>
       <h3>Learn and monitor subspaces</h3>
       <p>Use robust PCA, reconstruction diagnostics, bootstrap stability, and frozen-reference monitoring.</p>
     </a>
     <a class="gallery-card" href="workflows.html#features-embeddings-and-kernels">
       <div class="gallery-card-placeholder">Features<br>embeddings</div>
       <h3>Build robust feature geometry</h3>
       <p>Apply robust distances, whitening, kernels, and drift diagnostics to learned representations.</p>
     </a>
     <a class="gallery-card" href="workflows.html#structured-and-cellwise-data">
       <div class="gallery-card-placeholder">Cells<br>matrices</div>
       <h3>Handle structured contamination</h3>
       <p>Work with bad cells, missing values, matrix-valued observations, multilinear PCA, and sparse graphs.</p>
     </a>
     <a class="gallery-card" href="benchmark_gallery.html">
       <div class="gallery-card-placeholder">Evidence<br>limits</div>
       <h3>Review validation evidence</h3>
       <p>See task-specific benchmarks, failure cases, performance measurements, and reviewed external case studies.</p>
     </a>
   </div>

Choose your starting point
--------------------------

.. list-table:: Start from your immediate goal
   :header-rows: 1
   :widths: 30 32 38

   * - Goal
     - Start here
     - Typical first object
   * - Flag unusual rows in tabular data
     - :doc:`quickstart`
     - ``RobustOutlierDetector`` with ``FastMCD`` or ``RegularizedCauchy``; add ``ConformalAlertCalibrator`` for held-out score calibration
   * - Select a covariance or scatter estimator
     - :doc:`estimator_guide`
     - Match the estimator to rowwise, cellwise, heavy-tail, or high-dimensional contamination
   * - Reduce dimension and diagnose outliers
     - :doc:`workflows`
     - ``RobustPCA`` or ``CellPCA``
   * - Monitor a changing process or embedding stream
     - :doc:`monitoring`
     - ``RobustSubspaceMonitor`` or ``FeatureGeometry``
   * - Recover a sparse conditional-dependence graph
     - :doc:`sparse_precision`
     - ``RobustGraphicalLasso`` or ``SGLASSO``
   * - Work with matrix-valued or multilinear observations
     - :doc:`matrix_covariance`
     - ``MMCD`` or ``RobustMultilinearPCA``
   * - Compare methods and inspect evidence
     - :doc:`method_comparison` and :doc:`benchmark_gallery`
     - Use only benchmarks that match the fitted quantity and contamination model

The common pattern
------------------

.. code-block:: text

   observations, windows, or learned features
       -> robust location and scatter or robust low-rank fit
       -> covariance, precision, principal subspace, or latent structure
       -> scores, whitening, kernels, graphs, diagnostics, or monitoring

``robustcov`` is a numerical methods library, not a complete production anomaly
platform. It does not train neural networks, manage streaming infrastructure, or
choose a scientifically meaningful contamination model for you. See
:doc:`robust_geometry_layer` for the package boundaries and :doc:`api_stability`
for project maturity.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   robust_geometry_layer
   installation
   quickstart
   estimator_guide
   method_comparison
   api_stability
   faq

.. toctree::
   :maxdepth: 2
   :caption: Workflows

   workflows
   conformal_alert_calibration
   use_case_gallery

.. toctree::
   :maxdepth: 3
   :caption: Methods

   algorithms

.. toctree::
   :maxdepth: 3
   :caption: Examples and evidence

   benchmark_gallery
   external_data

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   diagnostics
   openmp
   methods_and_references
   robust_statistics_background
   references

.. toctree::
   :maxdepth: 1
   :caption: Project

   project_contributions

Project status
--------------

The project is in active alpha development. Core estimator interfaces and
fitted attributes are intended to remain recognizable, while newer monitoring,
geometry, and integration APIs may evolve before 1.0. Breaking changes and
deprecations are documented explicitly.

.. toctree::
   :maxdepth: 1
   :caption: Maintainer and extended material
   :hidden:

   notebooks
   kaggle_roadmap
   kaggle_examples
   external_demo_workflow
   legacy_external_examples
   release_readiness
   _generated/monte_carlo_summary
