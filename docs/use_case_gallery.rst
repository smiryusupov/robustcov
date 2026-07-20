Use-case gallery
================

The gallery is organized in two complementary ways:

* **Browse by method** when you already know whether you need ICA, PCA, a
  robust covariance estimator, or an anomaly-monitoring workflow.
* **Browse by application domain** when you want examples that resemble your
  data, such as finance, sensors, fraud, biomedical features, or embeddings.

Every detailed page shows the exact script to run.  The source is also embedded
at the bottom of the page, so the examples are visible in the documentation
rather than hidden in the repository.

Browse by method
----------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="gallery_methods/ica_source_separation.html">
       <img src="_static/gallery/sobi_source_separation/source_recovery.png" alt="ICA and SOBI source recovery">
       <h3>ICA and source separation</h3>
       <p>Independent components, temporally correlated sources, robust whitening, and BSS recovery metrics.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/pca_factor_models.html">
       <img src="_static/gallery/robust_factor_model/loading_recovery.png" alt="PCA and robust factor models">
       <h3>PCA and factor models</h3>
       <p>Robust PCA, cellwise PCA, multilinear PCA, factor scores, loadings, and low-rank reconstruction.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/robust_estimators.html">
       <div class="gallery-card-placeholder">Covariance<br>precision</div>
       <h3>Robust estimators</h3>
       <p>Covariance, scatter, precision, cellwise estimation, sparse graphs, and SPD geometry.</p>
     </a>
     <a class="gallery-card" href="gallery_methods/anomaly_monitoring.html">
       <div class="gallery-card-placeholder">Anomaly<br>monitoring</div>
       <h3>Anomaly detection and monitoring</h3>
       <p>Robust distances, one-class screening, sensor monitoring, embedding drift, and ML preprocessing.</p>
     </a>
   </div>

New latent-structure examples
-----------------------------

The ICA, SOBI, factor-model, and distribution-shift PCA additions have separate runnable examples:

.. code-block:: bash

   python examples/ica_two_scatter.py
   python examples/sobi_source_separation.py
   python examples/robust_factor_model.py
   python examples/distributionally_robust_pca.py
   python examples/distributionally_robust_pca_drift_monitoring.py

The scripts save figures under ``results/use_cases/<example-name>``.  The
checked-in gallery images are regenerated with:

.. code-block:: bash

   python docs/generate_gallery_assets.py --only ica_two_scatter sobi_source_separation robust_factor_model distributionally_robust_pca distributionally_robust_pca_drift_monitoring

Open the corresponding pages for explanation, captured output, plots, and
embedded source code:

* :doc:`Robust two-scatter ICA <gallery/ica_two_scatter>`
* :doc:`Robust SOBI <gallery/sobi_source_separation>`
* :doc:`Robust static factor model <gallery/robust_factor_model>`
* :doc:`Distributionally robust PCA <gallery/distributionally_robust_pca>`
* :doc:`DRO-PCA data-drift monitoring <gallery/distributionally_robust_pca_drift_monitoring>`

Browse by application domain
----------------------------

.. raw:: html

   <div class="gallery-grid compact-gallery-grid">
     <a class="gallery-card" href="gallery_topics/finance_and_risk.html">
       <div class="gallery-card-placeholder">Finance<br>risk</div>
       <h3>Finance and risk</h3>
       <p>Portfolio covariance, stress monitoring, yield-curve factors, and heavy-tailed returns.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/fraud_security_and_networks.html">
       <div class="gallery-card-placeholder">Fraud<br>security</div>
       <h3>Fraud, security, and networks</h3>
       <p>Fraud-like screening, suspicious records, and network-flow anomaly examples.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/sensors_industrial_quality.html">
       <div class="gallery-card-placeholder">Sensors<br>quality</div>
       <h3>Sensors and quality control</h3>
       <p>Sensor anomalies, predictive maintenance, process spectra, and multichannel windows.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/biomedical_images_embeddings.html">
       <div class="gallery-card-placeholder">Signals<br>embeddings</div>
       <h3>Biomedical, image, and embedding data</h3>
       <p>Feature-vector anomalies, representation drift, retrieval filtering, and multimodal screening.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/real_ml_datasets.html">
       <div class="gallery-card-placeholder">Real ML<br>datasets</div>
       <h3>Real ML datasets</h3>
       <p>Breast cancer, digits, and wine examples with baseline metrics.</p>
     </a>
     <a class="gallery-card" href="gallery_topics/ml_preprocessing.html">
       <div class="gallery-card-placeholder">ML<br>preprocess</div>
       <h3>Robust ML preprocessing</h3>
       <p>Clean, filter, or reweight contaminated observations before downstream models.</p>
     </a>
   </div>

Choose a method family
----------------------

.. list-table:: Method guide
   :header-rows: 1

   * - Goal
     - Open this group
     - Good first estimator
   * - Recover statistically independent latent signals
     - :doc:`ICA and source separation <gallery_methods/ica_source_separation>`
     - ``TwoScatterICA``
   * - Recover temporally correlated latent time series
     - :doc:`ICA and source separation <gallery_methods/ica_source_separation>`
     - ``RobustSOBI``
   * - Estimate low-rank principal components with robust diagnostics
     - :doc:`PCA and factor models <gallery_methods/pca_factor_models>`
     - ``RobustPCA``
   * - Decompose data into common factors and idiosyncratic residuals
     - :doc:`PCA and factor models <gallery_methods/pca_factor_models>`
     - ``RobustFactorModel``
   * - Protect a principal subspace against a stated train-to-deployment shift
     - :doc:`PCA and factor models <gallery_methods/pca_factor_models>`
     - experimental ``DistributionallyRobustPCA``
   * - Estimate robust covariance or scatter
     - :doc:`Robust estimators <gallery_methods/robust_estimators>`
     - ``FastMCD`` or ``RegularizedCauchy``
   * - Handle isolated corrupted cells and missing entries
     - :doc:`Robust estimators <gallery_methods/robust_estimators>`
     - ``CellMCD`` or ``CellRCov``
   * - Estimate a sparse conditional-dependence graph
     - :doc:`Robust estimators <gallery_methods/robust_estimators>`
     - ``RobustGraphicalLasso`` or ``SGLASSO``
   * - Rank anomalous rows or screen a one-class population
     - :doc:`Anomaly detection and monitoring <gallery_methods/anomaly_monitoring>`
     - ``RobustOutlierDetector``
   * - Monitor subspace or embedding drift
     - :doc:`Anomaly detection and monitoring <gallery_methods/anomaly_monitoring>`
     - ``RobustSubspaceMonitor`` or ``FeatureGeometry``
   * - Tolerate a stated covariance shift and alert on changes outside it
     - :doc:`Anomaly detection and monitoring <gallery_methods/anomaly_monitoring>`
     - experimental ``DistributionallyRobustPCA`` plus calibrated windows

Run examples by group
---------------------

Run the compact default set:

.. code-block:: bash

   python examples/run_use_case_gallery.py

Run a method family:

.. code-block:: bash

   python examples/run_use_case_gallery.py --group ica
   python examples/run_use_case_gallery.py --group pca
   python examples/run_use_case_gallery.py --group robust
   python examples/run_use_case_gallery.py --group monitoring

List the scripts in each group or run everything:

.. code-block:: bash

   python examples/run_use_case_gallery.py --list
   python examples/run_use_case_gallery.py --all

Regenerate documentation assets
-------------------------------

The gallery pages embed captured outputs and plots.  Refresh them after changing
examples:

.. code-block:: bash

   python docs/generate_gallery_assets.py
   sphinx-build -W --keep-going -b html docs docs/_build/html

Method-family pages
-------------------

.. toctree::
   :maxdepth: 2

   gallery_methods/ica_source_separation
   gallery_methods/pca_factor_models
   gallery_methods/robust_estimators
   gallery_methods/anomaly_monitoring

Application-domain pages
------------------------

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

   gallery/ica_two_scatter
   gallery/sobi_source_separation
   gallery/robust_factor_model
   gallery/distributionally_robust_pca
   gallery/distributionally_robust_pca_drift_monitoring
   gallery/robust_pca_yield_curve
   gallery/robust_pca_subspace_stability
   gallery/robust_pca_dependent_stability
   gallery/robust_pca_market_risk
   gallery/finance_risk
   gallery/portfolio_stress
   gallery/cellmcd_market_data
   gallery/cellrcov_high_dimensional
   gallery/robust_graphical_lasso_market_network
   gallery/spatial_sign_graphical_lasso
   gallery/cellpca_process_spectra
   gallery/sparse_cellpca_spectra
   gallery/fraud_screening
   gallery/network_traffic
   gallery/mmcd_sensor_windows
   gallery/robust_multilinear_pca
   gallery/sensor_anomaly
   gallery/maintenance_monitoring
   gallery/quality_control
   gallery/biomedical_signal
   gallery/image_feature_anomaly
   gallery/text_embedding_outliers
   gallery/robust_pca_embedding_monitoring
   gallery/density_power_pca
   gallery/robust_subspace_monitoring
   gallery/breast_cancer_screening
   gallery/digits_one_class
   gallery/wine_class_screening
   gallery/ml_preprocessing
   gallery/mrcd_high_dimensional_outliers
   gallery/kmrcd_nonlinear_manifold
   gallery/dets_detmm_tradeoff
   geometry
   gallery/feature_geometry_synthetic_ood
   gallery/feature_geometry_class_conditional_ood
   gallery/feature_geometry_drift_detection
   gallery/feature_geometry_similarity_kernel
   gallery/feature_geometry_mmd_diagnostic
   gallery/feature_geometry_embedding_monitoring
   gallery/gp_robust_input_metric
   gallery/embedding_reranking_robust_geometry
   gallery/multimodal_anomaly
