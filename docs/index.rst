robustcov documentation
=======================

``robustcov`` is a Python/C++ package for covariance and scatter estimation when
ordinary empirical covariance is too sensitive to a small part of the data.  It
includes robust estimators, Mahalanobis diagnostics, PCA, feature-space metrics,
SPD geometry, and rolling subspace monitoring.

The package is most useful when the data are heavy-tailed, contain leverage
points or outliers, or provide too few observations for an unrestricted sample
covariance matrix.

Where to start
--------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="use_case_gallery.html">
       <div class="gallery-card-placeholder">Use-case<br>gallery</div>
       <h3>Browse applications</h3>
       <p>Examples for finance, fraud, sensors, biomedical data, embeddings, and preprocessing.</p>
     </a>
     <a class="gallery-card" href="benchmark_gallery.html">
       <img src="_static/benchmarks/small_sample_rank.png" alt="Benchmark ranking plot">
       <h3>Review benchmark results</h3>
       <p>Accuracy under contamination, heavy-tail experiments, runtime comparisons, and OpenMP scaling.</p>
     </a>
     <a class="gallery-card" href="algorithms.html">
       <div class="gallery-card-placeholder">Math<br>and API</div>
       <h3>Read about the estimators</h3>
       <p>Assumptions, fitted quantities, references, and implementation notes for the main algorithms.</p>
     </a>
   </div>

A common workflow
-----------------

.. code-block:: text

   data or feature matrix
       -> robust location and scatter
       -> precision matrix or principal subspace
       -> distances, whitening, kernels, diagnostics, or monitoring

The main pieces are:

* ``FastMCD`` for sparse, separable contamination when the sample is larger than
  the feature dimension;
* regularized Cauchy, Student-t, and Tyler estimators for heavy tails and
  difficult covariance regimes;
* robust-distance plots and reports for anomaly analysis;
* ``RobustPCA`` for projection, reconstruction, and score/orthogonal-distance
  diagnostics;
* ``FeatureGeometry`` for robust distances and kernels on learned
  representations;
* ``RobustSubspaceMonitor`` for comparing rolling batches with a fixed
  reference period;
* SPD geometry and optional OpenMP acceleration for larger workloads.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   robust_geometry_layer
   installation
   quickstart
   estimator_guide
   use_case_gallery
   benchmark_gallery
   algorithms
   geometry
   robust_pca
   monitoring
   feature_geometry
   diagnostics
   openmp
   faq

.. toctree::
   :maxdepth: 2
   :caption: Reference and evidence

   api
   api_stability
   robust_statistics_background
   external_results_gallery
   references

Project status
--------------

The project is in active development.  The current estimators and examples are
tested, but public APIs may still change before a stable release.  See
:doc:`api_stability` for the compatibility policy.

.. toctree::
   :maxdepth: 1
   :caption: Extended material
   :hidden:

   notebooks
   kaggle_roadmap
   kaggle_examples
   external_demo_workflow
   release_readiness
