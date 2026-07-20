Examples by task and domain
===========================

Use this page as a directory, not as a complete list of every example. Choose
one route below, then browse the smaller set of examples on that landing page.
Individual example pages stay out of the global sidebar so the navigation
remains readable.

Browse by method
----------------

Choose this route when you already know the kind of statistical object you need.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="gallery_methods/ica_source_separation.html">
       <img src="_static/gallery/sobi_source_separation/source_recovery.png" alt="ICA and SOBI source recovery">
       <h3>ICA, SOBI, and source separation</h3>
       <p>Recover independent or temporally correlated latent signals with robust whitening and source-recovery diagnostics.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/pca_factor_models.html">
       <img src="_static/gallery/robust_factor_model/loading_recovery.png" alt="PCA and robust factor models">
       <h3>PCA and factor models</h3>
       <p>Learn robust low-rank structure, factors, loadings, reconstruction diagnostics, and monitored subspaces.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/robust_estimators.html">
       <div class="gallery-card-placeholder">Covariance<br>precision</div>
       <h3>Covariance, scatter, and precision</h3>
       <p>Estimate robust covariance, scatter, precision matrices, sparse graphs, and reusable distance geometry.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/anomaly_monitoring.html">
       <div class="gallery-card-placeholder">Anomaly<br>monitoring</div>
       <h3>Anomaly detection and monitoring</h3>
       <p>Rank unusual observations, screen one-class populations, and monitor sensors, subspaces, or embeddings.</p>
     </a>
   </div>

Browse by application domain
----------------------------

Choose this route when you want an example that resembles the data you work
with. The same example may appear in both a method family and a domain page.

.. raw:: html

   <div class="gallery-grid compact-gallery-grid">
     <a class="gallery-card" href="gallery_topics/finance_and_risk.html">
       <div class="gallery-card-placeholder">Finance<br>risk</div>
       <h3>Finance and risk</h3>
       <p>Portfolio covariance, stress monitoring, yield-curve factors, sparse networks, and heavy-tailed returns.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/fraud_security_and_networks.html">
       <div class="gallery-card-placeholder">Fraud<br>security</div>
       <h3>Fraud, security, and networks</h3>
       <p>Fraud-like screening, suspicious records, and multivariate network-flow anomalies.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/sensors_industrial_quality.html">
       <div class="gallery-card-placeholder">Sensors<br>quality</div>
       <h3>Sensors and quality control</h3>
       <p>Sensor anomalies, predictive maintenance, process spectra, multichannel windows, and degradation monitoring.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/biomedical_images_embeddings.html">
       <div class="gallery-card-placeholder">Signals<br>embeddings</div>
       <h3>Biomedical, image, and embedding data</h3>
       <p>Feature-vector anomalies, representation drift, retrieval filtering, and multimodal screening.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/real_ml_datasets.html">
       <div class="gallery-card-placeholder">Real ML<br>datasets</div>
       <h3>Real ML datasets</h3>
       <p>Small reproducible examples using breast-cancer, digits, wine, and multimodal data.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/ml_preprocessing.html">
       <div class="gallery-card-placeholder">ML<br>preprocess</div>
       <h3>Robust ML preprocessing</h3>
       <p>Clean, filter, reweight, or geometrically transform contaminated observations before downstream models.</p>
     </a>
   </div>

Good first examples
-------------------

These examples provide the shortest path from a common problem to a runnable
result. Use the category pages above when you need more specialized variants.

.. list-table:: Recommended starting points
   :header-rows: 1
   :widths: 30 38 32

   * - Goal
     - Start with
     - Main RobustCov role
   * - Rank unusual tabular observations
     - :doc:`Fraud-style anomaly screening <gallery/fraud_screening>`
     - Robust covariance and Mahalanobis scoring
   * - Estimate covariance under heavy tails
     - :doc:`Finance-style heavy-tail covariance <gallery/finance_risk>`
     - Regularized robust scatter
   * - Learn and monitor a changing subspace
     - :doc:`Rolling subspace monitoring <gallery/robust_subspace_monitoring>`
     - Robust PCA and frozen-reference monitoring
   * - Recover latent time series
     - :doc:`Robust SOBI <gallery/sobi_source_separation>`
     - Robust whitening and joint diagonalization
   * - Handle matrix-valued sensor windows
     - :doc:`Matrix MCD for sensor windows <gallery/mmcd_sensor_windows>`
     - Separable row/column covariance geometry
   * - Monitor learned representations
     - :doc:`Practical embedding monitoring <gallery/feature_geometry_embedding_monitoring>`
     - Robust feature distances and drift diagnostics

Run examples
------------

Run the compact default set or inspect the available groups:

.. code-block:: bash

   python examples/run_use_case_gallery.py
   python examples/run_use_case_gallery.py --list

Run one method family with ``--group ica``, ``--group pca``, ``--group robust``,
or ``--group monitoring``. See ``examples/README.md`` for asset-regeneration and
maintainer commands.

.. toctree::
   :maxdepth: 1
   :hidden:

   gallery_methods/ica_source_separation
   gallery_methods/pca_factor_models
   gallery_methods/robust_estimators
   gallery_methods/anomaly_monitoring
   gallery_topics/finance_and_risk
   gallery_topics/fraud_security_and_networks
   gallery_topics/sensors_industrial_quality
   gallery_topics/biomedical_images_embeddings
   gallery_topics/real_ml_datasets
   gallery_topics/ml_preprocessing
