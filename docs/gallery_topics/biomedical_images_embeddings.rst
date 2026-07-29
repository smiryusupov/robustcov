Biomedical, image, and embedding data
=====================================

These examples show robust-distance and robust-subspace diagnostics on feature
vectors extracted from signals, images, or embedding models.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/robust_subspace_monitoring.html">
       <img src="../_static/gallery/robust_subspace_monitoring/monitor_history.png" alt="Robust subspace stream monitoring">
       <h3>Frozen-reference stream monitoring</h3>
       <p>Decompose production drift into location, scatter, subspace, and record-level outlier signals.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_pca_embedding_monitoring.html">
       <img src="../_static/gallery/robust_pca_embedding_monitoring/batch_monitoring.png" alt="Robust PCA embedding monitoring">
       <h3>Production embedding monitoring</h3>
       <p>Separate drift inside the reference subspace from out-of-subspace production traffic.</p>
     </a>
     <a class="gallery-card" href="../gallery/biomedical_signal.html">
       <img src="../_static/gallery/biomedical_signal/distance_profile.png" alt="Biomedical signal windows">
       <h3>Biomedical signal windows</h3>
       <p>Detect abnormal signal windows from correlated time and frequency features.</p>
     </a>
     <a class="gallery-card" href="../gallery/image_feature_anomaly.html">
       <img src="../_static/gallery/image_feature_anomaly/distance_panel.png" alt="Image-feature anomaly">
       <h3>Image-feature anomaly</h3>
       <p>Find unusual feature vectors in image or representation spaces.</p>
     </a>
     <a class="gallery-card" href="../gallery/text_embedding_outliers.html">
       <img src="../_static/gallery/text_embedding_outliers/distance_panel.png" alt="Text embedding outliers">
       <h3>Text embedding outliers</h3>
       <p>Screen embedding vectors for off-topic or shifted observations.</p>
     </a>
   </div>

How to use this topic
---------------------

Start with production embedding monitoring when the system produces batches of
learned representations over time.  The remaining pages focus on point-level
robust-distance screening for particular feature domains.

Detailed pages
--------------

- :doc:`Frozen-reference stream monitoring <../gallery/robust_subspace_monitoring>`
- :doc:`Production embedding monitoring <../gallery/robust_pca_embedding_monitoring>`
- :doc:`Biomedical signal windows <../gallery/biomedical_signal>`
- :doc:`Image-feature anomaly <../gallery/image_feature_anomaly>`
- :doc:`Text embedding outliers <../gallery/text_embedding_outliers>`
