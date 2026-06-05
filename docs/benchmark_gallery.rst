Benchmark gallery
=================

The benchmark gallery is the main benchmark entry point.  It is designed for readers who want to
understand the evidence quickly: each card links to a focused benchmark page with plots, tables,
commands, and interpretation.

The gallery answers four practical questions:

* Which estimator works best for small-sample heavy-tailed covariance?
* How much faster is ``robustcov`` than common sklearn robust-covariance baselines?
* Does optional OpenMP parallelism help at larger scale?
* Where do robust covariance methods work well, and where do they fail?

Gallery cards
-------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="benchmarks/small_sample_heavy_tail.html">
       <img src="_static/benchmarks/small_sample_rank.png" alt="Small-sample heavy-tail ranking">
       <h3>Small-sample heavy-tail ranking</h3>
       <p>Regularized Cauchy, Student-t scatter, Tyler variants, MCD, Ledoit-Wolf, OAS, and empirical covariance compared across n, p, and tail weight.</p>
     </a>
     <a class="gallery-card" href="benchmarks/speed_comparison.html">
       <img src="_static/benchmarks/speed.png" alt="Speed comparison">
       <h3>Speed comparison</h3>
       <p>FastMCD and Tyler-family timing against sklearn covariance baselines in a representative contamination setting.</p>
     </a>
     <a class="gallery-card" href="benchmarks/openmp_scaling.html">
       <img src="_static/benchmarks/openmp_scaling.png" alt="OpenMP scaling">
       <h3>OpenMP scaling</h3>
       <p>Thread scaling for the C++ kernels used by FastMCD and RegularizedTyler.</p>
     </a>
     <a class="gallery-card" href="benchmarks/anomaly_baselines.html">
       <img src="_static/benchmarks/anomaly_baselines.png" alt="Anomaly baseline comparison">
       <h3>Anomaly detection baselines</h3>
       <p>Robust distance detectors compared with IsolationForest, LOF, OneClassSVM, and EllipticEnvelope.</p>
     </a>
     <a class="gallery-card" href="benchmarks/hard_contamination.html">
       <div class="gallery-card-placeholder">Hard<br>scenarios</div>
       <h3>Hard contamination scenarios</h3>
       <p>Mean shift, clustered contamination, variance contamination, leverage points, and heavy-tail inliers.</p>
     </a>
   </div>

Recommended benchmark workflow
------------------------------

Run the full report generator when you want the same assets used by the documentation:

.. code-block:: bash

   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python benchmarks/make_report.py --outdir results/report

This writes a standalone HTML report, Markdown report, CSV files, and plots.

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
