robustcov documentation
=======================

``robustcov`` is an experimental robust covariance, heavy-tail scatter, anomaly diagnostics, and benchmark-gallery package with a C++/pybind core.

The project is organized around two reader-friendly entry points:

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="use_case_gallery.html">
       <div class="gallery-card-placeholder">Use-case<br>gallery</div>
       <h3>Start from your application</h3>
       <p>Topic-based gallery: finance/risk, fraud/security, sensors/quality, biomedical/images/embeddings, real ML datasets, and preprocessing.</p>
     </a>
     <a class="gallery-card" href="benchmark_gallery.html">
       <img src="_static/benchmarks/small_sample_rank.png" alt="Benchmark ranking plot">
       <h3>Start from the evidence</h3>
       <p>Small-sample heavy-tail ranking, speed comparisons, OpenMP scaling, anomaly baselines, and hard contamination scenarios.</p>
     </a>
     <a class="gallery-card" href="algorithms.html">
       <div class="gallery-card-placeholder">Math<br>and API</div>
       <h3>Understand the estimators</h3>
       <p>FastMCD, Tyler shape, Regularized Tyler, Student-t scatter, Cauchy scatter, diagnostics, and references.</p>
     </a>
   </div>

Core ideas
----------

* ``FastMCD`` gives efficient classical robust covariance for separable contamination when ``n`` is comfortably larger than ``p``.
* ``RegularizedCauchy`` and ``StudentTScatter`` target small-sample, high-dimensional, heavy-tailed covariance problems.
* Robust-distance diagnostics turn fitted estimators into interpretable anomaly scores, profiles, QQ plots, and reports.
* Optional OpenMP acceleration improves larger workloads and benchmark/report generation.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   estimator_guide
   use_case_gallery
   benchmark_gallery
   algorithms
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

Why not focus on MVE?
---------------------

Minimum-volume ellipsoid estimators are historically important, but the benchmark evidence in this project points to a stronger niche: efficient MCD for separable outliers and regularized heavy-tail scatter for small samples. MVE may become an experimental add-on later, but it is not currently the core differentiator.
