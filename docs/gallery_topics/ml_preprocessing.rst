Robust ML preprocessing
=======================

This topic is for users who do not want anomaly detection as the final task, but want to clean or score data before fitting another model.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/ml_preprocessing.html">
       <img src="../_static/gallery/ml_preprocessing/accuracy_comparison.png" alt="Robust ML preprocessing">
       <h3>Robust ML preprocessing</h3>
       <p>Remove suspicious training rows before fitting a downstream classifier.</p>
     </a>
     <a class="gallery-card" href="../gallery/gp_robust_input_metric.html">
       <img src="../_static/gallery/gp_robust_input_metric/kernel_comparison.png" alt="Robust GP kernel input metric">
       <h3>Robust GP kernel / input metric</h3>
       <p>Use robust covariance as input-space geometry for existing GP kernels.</p>
     </a>
     <a class="gallery-card" href="../gallery/mrcd_high_dimensional_outliers.html">
       <img src="../_static/gallery/mrcd_high_dimensional_outliers/distance_crossplot.png" alt="MRCD high-dimensional row outliers">
       <h3>High-dimensional row contamination</h3>
       <p>Estimate a well-conditioned covariance when the feature count exceeds the sample size.</p>
     </a>
     <a class="gallery-card" href="../gallery/kmrcd_nonlinear_manifold.html">
       <img src="../_static/gallery/kmrcd_nonlinear_manifold/kernel_distance_contours.png" alt="Kernel MRCD curved-manifold outlier detection">
       <h3>Nonlinear robust distances</h3>
       <p>Compare linear MRCD with an RBF kernel on a curved majority structure.</p>
     </a>
   </div>

How to use this topic
---------------------

Start with the first card if you want the simplest demonstration.  Then move to the more specialized page when the data shape matches your problem.  Every page includes captured output, plots, interpretation notes, and a command to reproduce the result.

Detailed pages
--------------

- :doc:`ML preprocessing <../gallery/ml_preprocessing>`
- :doc:`Embedding reranking with robust geometry <../gallery/embedding_reranking_robust_geometry>`