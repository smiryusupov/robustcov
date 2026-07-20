Anomaly detection and monitoring
================================

Use this section when the final task is ranking unusual observations,
screening a one-class population, or monitoring drift over time.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/fraud_screening.html">
       <img src="../_static/gallery/fraud_screening/distance_profile.png" alt="Fraud anomaly distance profile">
       <h3>Tabular anomaly screening</h3>
       <p>Rank suspicious records with robust distances and contamination-aware thresholds.</p>
     </a>
     <a class="gallery-card" href="../gallery/sensor_anomaly.html">
       <div class="gallery-card-placeholder">Sensors<br>anomalies</div>
       <h3>Sensor anomalies</h3>
       <p>Detect multivariate bursts while preserving correlated process structure.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_subspace_monitoring.html">
       <div class="gallery-card-placeholder">Subspace<br>monitoring</div>
       <h3>Subspace monitoring</h3>
       <p>Monitor score-space and orthogonal drift against a robust reference period.</p>
     </a>
     <a class="gallery-card" href="../gallery/distributionally_robust_pca_drift_monitoring.html">
       <img src="../_static/gallery/distributionally_robust_pca_drift_monitoring/drift_timeline.png" alt="Distributionally robust PCA data-drift monitor">
       <h3>DRO-PCA drift monitoring</h3>
       <p>Tolerate anticipated covariance shift while alerting on off-geometry drift.</p>
     </a>
     <a class="gallery-card" href="../gallery/feature_geometry_embedding_monitoring.html">
       <div class="gallery-card-placeholder">Embeddings<br>drift</div>
       <h3>Embedding monitoring</h3>
       <p>Track robust feature geometry and representation drift in production batches.</p>
     </a>
     <a class="gallery-card" href="../gallery/breast_cancer_screening.html">
       <div class="gallery-card-placeholder">Real data<br>screening</div>
       <h3>Real ML datasets</h3>
       <p>Compare robust anomaly rankings with standard baselines on reproducible datasets.</p>
     </a>
     <a class="gallery-card" href="../gallery/ml_preprocessing.html">
       <div class="gallery-card-placeholder">ML<br>preprocessing</div>
       <h3>Robust preprocessing</h3>
       <p>Filter or weight contaminated rows before downstream supervised learning.</p>
     </a>
   </div>

Runnable examples
-----------------

.. code-block:: bash

   python examples/use_case_fraud_screening.py
   python examples/use_case_sensor_anomaly.py
   python examples/plot_robust_subspace_monitoring.py
   python examples/distributionally_robust_pca_drift_monitoring.py
   python examples/feature_geometry_embedding_monitoring.py
   python examples/run_use_case_gallery.py --group monitoring

Detailed pages
--------------

.. toctree::
   :maxdepth: 1

   ../gallery/fraud_screening
   ../gallery/network_traffic
   ../gallery/sensor_anomaly
   ../gallery/maintenance_monitoring
   ../gallery/quality_control
   ../gallery/robust_subspace_monitoring
   ../gallery/distributionally_robust_pca_drift_monitoring
   ../gallery/feature_geometry_drift_detection
   ../gallery/feature_geometry_embedding_monitoring
   ../gallery/breast_cancer_screening
   ../gallery/digits_one_class
   ../gallery/wine_class_screening
   ../gallery/ml_preprocessing
