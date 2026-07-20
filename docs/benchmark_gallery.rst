Benchmark gallery
=================

The benchmark gallery is the main benchmark entry point.  It is designed for readers who want to
understand the evidence quickly: each card links to a focused benchmark page with plots, tables,
commands, and interpretation.

The gallery and method-comparison page answer seven practical questions:

* Which estimator matches the contamination model and dimensional regime?
* Which estimator works best for small-sample heavy-tailed covariance?
* How do the current covariance and scatter estimators compare in speed across low-dimensional, heavy-tail, and high-dimensional workloads?
* Which complete estimators and native kernels actually use OpenMP, and how do they scale?
* Where do robust covariance methods work well, and where do they fail?
* How well do robust ICA and SOBI recover latent sources under impulsive contamination?
* How accurately do robust PCA and robust factor models recover low-rank structure?
* Does weighted-Wasserstein PCA improve held-out reconstruction under a stated distribution shift, and what does it cost when no shift occurs?

Gallery cards
-------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="method_comparison.html">
       <div class="gallery-card-placeholder">Choose<br>a method</div>
       <h3>Method comparison</h3>
       <p>Capability tables and task-specific benchmarks for rowwise, cellwise, high-dimensional, matrix, PCA, and sparse-graph problems.</p>
     </a>
     <a class="gallery-card" href="benchmarks/small_sample_heavy_tail.html">
       <img src="_static/benchmarks/small_sample_rank.png" alt="">
       <h3>Small-sample rowwise covariance and scatter ranking</h3>
       <p>The complete relevant catalog—including MRCD, DetS/DetMM, Tyler and M-scatter variants, MCD, shrinkage baselines, and the automatic selector—ranked only where each method is mathematically applicable.</p>
     </a>
     <a class="gallery-card" href="benchmarks/speed_comparison.html">
       <img src="_static/benchmarks/speed.png" alt="">
       <h3>Rowwise covariance speed by workload</h3>
       <p>Complete-fit timing for the shared covariance/scatter catalog on row contamination, heavy tails, and p &gt; n workloads, with inapplicable methods reported explicitly.</p>
     </a>
     <a class="gallery-card" href="benchmarks/openmp_scaling.html">
       <img src="_static/benchmarks/openmp_scaling.png" alt="">
       <h3>Threaded native scaling</h3>
       <p>OpenMP scaling and numerical drift for FastMCD, Tyler fits, vector/matrix Mahalanobis batches, and weighted Tucker score solves.</p>
     </a>
     <a class="gallery-card" href="benchmarks/anomaly_baselines.html">
       <img src="_static/benchmarks/anomaly_baselines.png" alt="">
       <h3>Anomaly detection baselines</h3>
       <p>Robust distance detectors compared with IsolationForest, LOF, OneClassSVM, and EllipticEnvelope.</p>
     </a>
     <a class="gallery-card" href="benchmarks/hard_contamination.html">
       <div class="gallery-card-placeholder">Hard<br>scenarios</div>
       <h3>Hard contamination scenarios</h3>
       <p>Mean shift, clustered contamination, variance contamination, leverage points, and heavy-tail inliers.</p>
     </a>
     <a class="gallery-card" href="benchmarks/latent_structure.html">
       <img src="_static/benchmarks/latent_structure/sobi_mdi.png" alt="">
       <h3>ICA, SOBI, PCA, and factor models</h3>
       <p>Permutation-aware source recovery, robust subspace estimation, factor-count selection, common-component reconstruction, and complete-fit timing.</p>
     </a>
     <a class="gallery-card" href="benchmarks/distributionally_robust_pca.html">
       <img src="_static/benchmarks/distributionally_robust_pca.png" alt="">
       <h3>Distributionally robust PCA</h3>
       <p>Held-out target reconstruction under structured covariance shift, no-shift efficiency, and contamination-only controls.</p>
     </a>
     <a class="gallery-card" href="benchmark_inventory.html">
       <div class="gallery-card-placeholder">Coverage<br>inventory</div>
       <h3>Benchmark coverage inventory</h3>
       <p>Maps each canonical public estimator to comparative evidence, validation gates, performance gates, or an end-to-end workflow.</p>
     </a>
   </div>

Recommended benchmark workflow
------------------------------

Run the task-specific comparison when choosing among the current estimator families:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
   python benchmarks/compare_methods.py \
       --profile quick \
       --csv results/method_comparison.csv

See :doc:`method_comparison` for the benchmark design and interpretation.  Run
the combined report generator when you want the expanded covariance speed, threaded-native, anomaly-baseline,
heavy-tail, and latent-structure gallery assets:

.. code-block:: bash

   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python benchmarks/make_report.py --outdir results/report

This writes a standalone HTML report, Markdown report, CSV files, and plots.

Run the latent-structure suite separately when evaluating ICA, SOBI, robust
PCA, or factor models:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
   python benchmarks/latent_structure_benchmarks.py \
       --profile quick \
       --csv results/latent_structure.csv \
       --plot-dir results/latent_structure_plots

Run the distribution-shift benchmark independently because its primary metric
is held-out target risk rather than contamination recovery:

.. code-block:: bash

   python benchmarks/distributionally_robust_pca.py \
       --profile quick \
       --csv results/distributionally_robust_pca.csv \
       --plot results/distributionally_robust_pca.png

.. code-block:: text

   results/report/benchmark_report.html
   results/report/benchmark_report.md
   results/report/small_sample.csv
   results/report/small_sample_summary.csv
   results/report/small_sample_rank.png
   results/report/speed.csv
   results/report/speed.png
   results/report/openmp_scaling.csv
   results/report/openmp_scaling.png
   results/report/anomaly_baselines.csv
   results/report/anomaly_baselines.png
   results/report/hard_scenarios.csv

How to read the gallery
-----------------------

A single benchmark row is rarely enough.  Prefer rank summaries, median error, win rate, and
scenario-specific interpretation.  ``RegularizedCauchy`` is usually the strongest small-sample
heavy-tail covariance estimator.  ``FastMCD`` is the classical choice for separable contamination
when the uncontaminated majority is well defined.  ``RegularizedTyler`` is best described as a
robust shape estimator and should not be advertised as the universal covariance-recovery winner.

Detailed benchmark pages
------------------------

.. toctree::
   :maxdepth: 1
   :hidden:

   benchmarks/small_sample_heavy_tail
   benchmarks/speed_comparison
   benchmarks/openmp_scaling
   benchmarks/anomaly_baselines
   benchmarks/hard_contamination
   benchmarks/latent_structure
   benchmarks/distributionally_robust_pca
   benchmark_inventory
