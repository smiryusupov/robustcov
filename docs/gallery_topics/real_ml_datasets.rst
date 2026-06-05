Real ML datasets
================

These examples use built-in scikit-learn datasets so the results are reproducible without downloads.  They compare robustcov scores against familiar anomaly baselines.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/breast_cancer_screening.html">
       <img src="../_static/gallery/breast_cancer_screening/baseline_f1.png" alt="Breast-cancer screening">
       <h3>Breast-cancer screening</h3>
       <p>Diagnostic tabular anomaly screening with baseline comparison.</p>
     </a>
     <a class="gallery-card" href="../gallery/digits_one_class.html">
       <img src="../_static/gallery/digits_one_class/baseline_f1.png" alt="Digits one-class anomaly">
       <h3>Digits one-class anomaly</h3>
       <p>PCA image features with robust distances and anomaly baselines.</p>
     </a>
     <a class="gallery-card" href="../gallery/wine_class_screening.html">
       <img src="../_static/gallery/wine_class_screening/baseline_f1.png" alt="Wine class screening">
       <h3>Wine class screening</h3>
       <p>Correlated chemical measurements with robust ensemble scores.</p>
     </a>

     <a class="gallery-card" href="../gallery/multimodal_anomaly.html">
       <img src="../_static/gallery/multimodal_anomaly/cluster_distance_panel.png" alt="Multimodal anomaly detection">
       <h3>Multimodal anomaly detection</h3>
       <p>Cluster-aware robust distances for datasets with several valid modes.</p>
     </a>
   </div>

How to use this topic
---------------------

Start with the first card if you want the simplest demonstration.  Then move to the more specialized page when the data shape matches your problem.  Every page includes captured output, plots, interpretation notes, and a command to reproduce the result.

Detailed pages
--------------

.. toctree::
   :maxdepth: 1

   ../gallery/breast_cancer_screening
   ../gallery/digits_one_class
   ../gallery/wine_class_screening
   ../gallery/multimodal_anomaly
