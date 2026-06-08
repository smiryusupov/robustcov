Use-case gallery
================

The use-case gallery is organized by problem theme.  This is usually the best way to enter the documentation: choose the topic that looks like your application, then open a card with plots, captured output, and interpretation.

Start here
----------

If you are new to the package, start with one of these examples:

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="gallery_topics/fraud_security_and_networks.html">
       <img src="_static/gallery/fraud_screening/distance_profile.png" alt="Fraud robust-distance profile">
       <h3>Fraud and security</h3>
       <p>Suspicious tabular records, network traffic anomalies, and robust distance ranking.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/finance_and_risk.html">
       <img src="_static/gallery/finance_risk/covariance.png" alt="Finance covariance heatmap">
       <h3>Finance and risk</h3>
       <p>Heavy-tailed covariance, portfolio stress, and market-risk style examples.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/real_ml_datasets.html">
       <img src="_static/gallery/digits_one_class/baseline_f1.png" alt="Real ML baseline comparison">
       <h3>Real ML datasets</h3>
       <p>Breast cancer, digits, and wine examples with sklearn baseline comparisons.</p>
     </a>
     <a class="gallery-card" href="geometry.html">
       <img src="_static/examples/spd_geometry_drift_monitoring.png" alt="SPD geometry covariance drift monitoring">
       <h3>Scatter geometry</h3>
       <p>Compare covariance and scatter matrices with affine-invariant and log-Euclidean geometry.</p>
     </a>
     <a class="gallery-card" href="gallery/gp_robust_input_metric.html">
       <div class="gallery-card-placeholder">Kernels<br>GP metrics</div>
       <h3>Similarity and kernels</h3>
       <p>Use robust scatter estimates as input metrics for kernels, GP workflows, and similarity methods.</p>
     </a>
   </div>

Browse by topic
---------------

.. raw:: html

   <div class="gallery-grid compact-gallery-grid">
     <a class="gallery-card" href="gallery_topics/finance_and_risk.html">
       <div class="gallery-card-placeholder">Finance<br>risk</div>
       <h3>Finance and risk</h3>
       <p>Portfolio covariance, stress monitoring, and heavy-tailed returns.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/fraud_security_and_networks.html">
       <div class="gallery-card-placeholder">Fraud<br>security</div>
       <h3>Fraud, security, and networks</h3>
       <p>Fraud-like screening and network-flow anomaly examples.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/sensors_industrial_quality.html">
       <div class="gallery-card-placeholder">Sensors<br>quality</div>
       <h3>Sensors and quality control</h3>
       <p>Sensor anomaly, predictive maintenance, and process monitoring.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/biomedical_images_embeddings.html">
       <div class="gallery-card-placeholder">Signals<br>embeddings</div>
       <h3>Biomedical, image, and embedding data</h3>
       <p>Feature-vector anomaly detection for signals, images, and embeddings.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/real_ml_datasets.html">
       <div class="gallery-card-placeholder">Real ML<br>datasets</div>
       <h3>Real ML datasets</h3>
       <p>Reproducible built-in datasets with baseline metrics and plots.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/ml_preprocessing.html">
       <div class="gallery-card-placeholder">ML<br>preprocess</div>
       <h3>Robust ML preprocessing</h3>
       <p>Use robust distances before downstream classification.</p>
     </a>
          <a class="gallery-card" href="geometry.html">
       <div class="gallery-card-placeholder">SPD<br>geometry</div>
       <h3>Scatter geometry</h3>
       <p>SPD matrix distances, geodesics, covariance drift, and robust shape comparison.</p>
     </a>
     <a class="gallery-card" href="gallery/gp_robust_input_metric.html">
       <div class="gallery-card-placeholder">Kernels<br>similarity</div>
       <h3>Similarity, kernels, and GP metrics</h3>
       <p>Robust input metrics for kernel methods, Gaussian-process workflows, and embedding retrieval.</p>
     </a>
   </div>

Which topic should I open?
--------------------------

.. list-table:: Topic guide
   :header-rows: 1

   * - Problem you have
     - Open this topic
     - Good first estimator
   * - Fraud, suspicious transactions, unusual records
     - Fraud, security, and networks
     - ``FastMCD`` or ``AutoRobustAnomalyDetector``
   * - Portfolio, returns, covariance, stress periods
     - Finance and risk
     - ``RegularizedCauchy``
   * - Sensors, process drift, industrial faults
     - Sensors and quality control
     - ``FastMCD`` or ``RegularizedCauchy``
   * - Signal/image/embedding feature vectors
     - Biomedical, image, and embedding data
     - ``AutoRobustScatter`` or ``RegularizedCauchy``
   * - Reproducible ML benchmark examples
     - Real ML datasets
     - ``RobustOutlierDetector`` with baseline comparison
   * - Clean training data before a classifier
     - Robust ML preprocessing
     - robust-distance filtering
   * - Compare covariance/scatter matrices or monitor covariance drift
     - Scatter geometry
     - ``robustcov.geometry`` with fitted scatter estimates
   * - Build robust similarity, kernel, or GP input metrics
     - Similarity, kernels, and GP metrics
     - robust input metric from ``RegularizedCauchy`` or ``FastMCD``

Run the gallery
---------------

.. code-block:: bash

   python examples/run_use_case_gallery.py

Run every gallery script:

.. code-block:: bash

   python examples/run_use_case_gallery.py --all

Regenerate documentation assets
-------------------------------

The gallery pages embed captured outputs and plots.  Refresh them after changing examples:

.. code-block:: bash

   python docs/generate_gallery_assets.py
   sphinx-build -b html docs docs/_build/html

Topic pages
-----------

.. toctree::
   :maxdepth: 2

   gallery_topics/finance_and_risk
   gallery_topics/fraud_security_and_networks
   gallery_topics/sensors_industrial_quality
   gallery_topics/biomedical_images_embeddings
   gallery_topics/real_ml_datasets
   gallery_topics/ml_preprocessing

All detailed pages
------------------

.. toctree::
   :maxdepth: 1

   gallery/finance_risk
   gallery/portfolio_stress
   gallery/fraud_screening
   gallery/network_traffic
   gallery/sensor_anomaly
   gallery/maintenance_monitoring
   gallery/quality_control
   gallery/biomedical_signal
   gallery/image_feature_anomaly
   gallery/text_embedding_outliers
   gallery/breast_cancer_screening
   gallery/digits_one_class
   gallery/wine_class_screening
   gallery/ml_preprocessing
   geometry
   gallery/gp_robust_input_metric
   gallery/embedding_reranking_robust_geometry
   gallery/multimodal_anomaly
