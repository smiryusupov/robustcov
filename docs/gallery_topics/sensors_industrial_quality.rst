Sensors, industrial monitoring, and quality control
===================================================

These examples focus on process monitoring: sensor streams, maintenance windows, and manufacturing-style quality-control data.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/mmcd_sensor_windows.html">
       <img src="../_static/gallery/mmcd_sensor_windows/contribution_heatmap.png" alt="MMCD matrix-distance contributions">
       <h3>Matrix-valued sensor windows</h3>
       <p>Estimate separate sensor and within-window covariance factors without flattening each observation.</p>
     </a>
     <a class="gallery-card" href="../gallery/cellpca_process_spectra.html">
       <img src="../_static/gallery/cellpca_process_spectra/residual_cellmap.png" alt="CellPCA residual cellmap">
       <h3>Process spectra with bad cells</h3>
       <p>Fit a low-rank process model while separating bad wavelengths from abnormal batches.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_subspace_monitoring.html">
       <img src="../_static/gallery/robust_subspace_monitoring/monitor_history.png" alt="Robust rolling subspace monitoring">
       <h3>Rolling subspace monitoring</h3>
       <p>Use a frozen robust reference to distinguish location, covariance, and factor drift.</p>
     </a>
     <a class="gallery-card" href="../gallery/sensor_anomaly.html">
       <img src="../_static/gallery/sensor_anomaly/distance_panel.png" alt="Sensor anomaly detection">
       <h3>Sensor anomaly detection</h3>
       <p>Detect unusual sensor windows using robust covariance geometry.</p>
     </a>
     <a class="gallery-card" href="../gallery/maintenance_monitoring.html">
       <img src="../_static/gallery/maintenance_monitoring/time_profile.png" alt="Predictive maintenance">
       <h3>Predictive maintenance</h3>
       <p>Flag degradation-like windows before downstream rules or classifiers.</p>
     </a>
     <a class="gallery-card" href="../gallery/quality_control.html">
       <img src="../_static/gallery/quality_control/support_ellipse.png" alt="Quality control">
       <h3>Quality control</h3>
       <p>Separate stable process variation from abnormal production runs.</p>
     </a>
   </div>

How to use this topic
---------------------

Start with the first card if you want the simplest demonstration.  Then move to the more specialized page when the data shape matches your problem.  Every page includes captured output, plots, interpretation notes, and a command to reproduce the result.

Detailed pages
--------------

- :doc:`Rolling subspace monitoring <../gallery/robust_subspace_monitoring>`
- :doc:`Matrix-valued sensor windows <../gallery/mmcd_sensor_windows>`
- :doc:`Process spectra with cellwise and casewise contamination <../gallery/cellpca_process_spectra>`
- :doc:`Sensor anomaly <../gallery/sensor_anomaly>`
- :doc:`Maintenance monitoring <../gallery/maintenance_monitoring>`
- :doc:`Quality control <../gallery/quality_control>`