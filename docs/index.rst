.. title:: RobustCov
.. meta::
   :description: Robust covariance, anomaly scoring, PCA, decomposition, and monitoring for difficult multivariate data.

.. raw:: html

   <section class="robustcov-hero" aria-labelledby="robustcov-hero-title">
     <div class="robustcov-hero-brand">
       <img src="_static/brand/robustcov-mark.png" alt="RobustCov covariance ellipse and outlier mark">
       <div class="robustcov-hero-copy">
         <h1 id="robustcov-hero-title" class="robustcov-wordmark"><span class="robustcov-wordmark-robust">robust</span><span class="robustcov-wordmark-cov">cov</span></h1>
         <p class="robustcov-tagline">Robust multivariate geometry for difficult data.</p>
       </div>
     </div>
     <p class="robustcov-hero-summary">
       Estimate covariance and precision, score anomalies, decompose corrupted matrices,
       learn robust subspaces, and monitor distribution change when ordinary covariance
       is unreliable.
     </p>
     <div class="robustcov-hero-actions">
       <a class="robustcov-button robustcov-button-primary" href="quickstart.html">Start with the quickstart</a>
       <a class="robustcov-button" href="estimator_guide.html">Choose a method</a>
       <a class="robustcov-button" href="use_case_gallery.html">Browse examples</a>
     </div>
     <div class="robustcov-hero-proof" aria-label="Package characteristics">
       <span>sklearn-style APIs</span>
       <span>NumPy / SciPy</span>
       <span>optional C++ / OpenMP</span>
       <span>explicit assumptions and evidence</span>
     </div>
   </section>

Choose a path
-------------

.. raw:: html

   <div class="robustcov-path-grid">
     <a class="robustcov-path-card" href="quickstart.html">
       <span class="robustcov-card-number">01</span>
       <h3>Use the package</h3>
       <p>Fit a robust geometry, score observations, and inspect the fitted diagnostics in a few lines.</p>
       <strong>Quickstart →</strong>
     </a>
     <a class="robustcov-path-card" href="estimator_guide.html">
       <span class="robustcov-card-number">02</span>
       <h3>Choose the right model</h3>
       <p>Start from row outliers, bad cells, heavy tails, high dimensions, matrices, or a changing stream.</p>
       <strong>Estimator guide →</strong>
     </a>
     <a class="robustcov-path-card" href="benchmark_gallery.html">
       <span class="robustcov-card-number">03</span>
       <h3>Inspect the evidence</h3>
       <p>Review task-specific benchmarks, known limitations, performance checks, and C-MAPSS case studies.</p>
       <strong>Benchmarks →</strong>
     </a>
   </div>

What RobustCov provides
-----------------------

.. raw:: html

   <div class="robustcov-capability-grid">
     <a class="robustcov-capability" href="workflows.html#covariance-scatter-and-anomaly-scores">
       <span class="robustcov-capability-icon">Σ</span>
       <h3>Estimate geometry</h3>
       <p>Robust location, covariance, scatter, precision, Mahalanobis scores, whitening, and kernels.</p>
     </a>
     <a class="robustcov-capability" href="workflows.html#pca-subspaces-and-monitoring">
       <span class="robustcov-capability-icon">L + S</span>
       <h3>Decompose and reduce</h3>
       <p>Robust PCA, low-rank-plus-sparse decomposition, cellwise PCA, stability, and latent factors.</p>
     </a>
     <a class="robustcov-capability" href="monitoring.html">
       <span class="robustcov-capability-icon">p ≤ α</span>
       <h3>Detect and monitor</h3>
       <p>Anomaly diagnostics, conformal alert calibration, frozen references, and adaptive subspace tracking.</p>
     </a>
     <a class="robustcov-capability" href="workflows.html#structured-and-cellwise-data">
       <span class="robustcov-capability-icon">X₁…Xₙ</span>
       <h3>Handle difficult structure</h3>
       <p>Bad cells, missing entries, high-dimensional tables, matrices, tensors, embeddings, and sparse graphs.</p>
     </a>
   </div>

A 60-second start
-----------------

.. code-block:: python

   import numpy as np
   import robustcov as rc

   rng = np.random.default_rng(0)
   X = rng.standard_t(df=3, size=(400, 5))
   X[:30] += 8.0

   detector = rc.RobustOutlierDetector(
       estimator=rc.FastMCD(quality="balanced", random_state=42),
       contamination=0.075,
   ).fit(X)

   unusual_rows = np.flatnonzero(detector.labels_ == -1)
   robust_distances = detector.mahalanobis(X)

See :doc:`quickstart` for held-out conformal calibration, native availability,
and a complete fitted-object walkthrough.

Choose by data problem
----------------------

.. list-table:: A practical first choice
   :header-rows: 1
   :widths: 31 35 34

   * - Data problem
     - Start with
     - Read next
   * - A minority of complete rows are outliers
     - ``FastMCD``, ``DetS``, or ``DetMM``
     - :doc:`estimator_guide`
   * - Heavy tails or an ill-conditioned / high-dimensional covariance
     - ``RegularizedCauchy``, ``StudentTScatter``, ``RegularizedTyler``, or ``MRCD``
     - :doc:`method_comparison`
   * - Isolated bad cells or missing entries
     - ``CellMCD``, ``CellRCov``, ``CellPCA``, or ``SparseCellPCA``
     - :doc:`workflows`
   * - One matrix is low rank plus sparse gross corruption
     - ``PrincipalComponentPursuit`` / ``PCP``
     - :doc:`principal_component_pursuit`
   * - A reference geometry must be monitored over time
     - ``RobustSubspaceMonitor``, ``ConformalAlertCalibrator``, or experimental ``OnlineRobustSubspaceTracker``
     - :doc:`monitoring`
   * - Observations are matrices, tensors, or learned embeddings
     - ``MMCD``, ``RobustMultilinearPCA``, or ``FeatureGeometry``
     - :doc:`use_case_gallery`

.. raw:: html

   <div class="robustcov-scope-note">
     <strong>Scope.</strong> RobustCov is a numerical methods library, not a complete
     production anomaly platform. It does not train neural networks, operate data
     pipelines, or select a scientifically meaningful contamination model for you.
     The method pages state assumptions, API maturity, and evidence boundaries.
   </div>

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
   online_subspace_tracking
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
