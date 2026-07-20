PCA and factor models
=====================

Use this section for low-rank structure, interpretable factors, reconstruction,
and score-versus-orthogonal-distance diagnostics.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/robust_pca_yield_curve.html">
       <img src="../_static/gallery/robust_pca_yield_curve/factor_loadings.png" alt="Robust PCA yield curve factors">
       <h3>Robust PCA factors</h3>
       <p>Recover stable level, slope, and curvature factors under contaminated curve observations.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_factor_model.html">
       <img src="../_static/gallery/robust_factor_model/loading_recovery.png" alt="Robust factor loading recovery">
       <h3>Robust factor model</h3>
       <p>Estimate factor count, loadings, scores, common components, and idiosyncratic residuals.</p>
     </a>
     <a class="gallery-card" href="../gallery/distributionally_robust_pca.html">
       <img src="../_static/gallery/distributionally_robust_pca/target_risk.png" alt="Distributionally robust PCA held-out target risk">
       <h3>Distributionally robust PCA</h3>
       <p>Protect a principal subspace against a stated weighted-Wasserstein covariance-shift geometry.</p>
     </a>
     <a class="gallery-card" href="../gallery/distributionally_robust_pca_drift_monitoring.html">
       <img src="../_static/gallery/distributionally_robust_pca_drift_monitoring/drift_timeline.png" alt="Distributionally robust PCA drift monitoring timeline">
       <h3>DRO-PCA drift monitoring</h3>
       <p>Calibrate window alerts while tolerating anticipated geometry-aligned covariance shift.</p>
     </a>
     <a class="gallery-card" href="../gallery/cellpca_process_spectra.html">
       <img src="../_static/gallery/cellpca_process_spectra/residual_cellmap.png" alt="CellPCA process spectra">
       <h3>Cellwise PCA</h3>
       <p>Fit a low-rank table when individual entries, rows, and missing values are all present.</p>
     </a>
     <a class="gallery-card" href="../gallery/sparse_cellpca_spectra.html">
       <img src="../_static/gallery/sparse_cellpca_spectra/sparse_loadings.png" alt="Sparse CellPCA loading paths">
       <h3>Sparse CellPCA</h3>
       <p>Produce interpretable sparse loadings for contaminated process spectra.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_multilinear_pca.html">
       <div class="gallery-card-placeholder">Matrix PCA<br>two modes</div>
       <h3>Robust multilinear PCA</h3>
       <p>Model matrix-valued observations with robust row and column subspaces.</p>
     </a>
     <a class="gallery-card" href="../gallery/density_power_pca.html">
       <div class="gallery-card-placeholder">DP-PCA<br>mixed contamination</div>
       <h3>Density-power PCA</h3>
       <p>Direct robust low-rank fitting under rowwise and cellwise contamination.</p>
     </a>
   </div>

Runnable examples
-----------------

.. code-block:: bash

   python examples/distributionally_robust_pca.py
   python examples/distributionally_robust_pca_drift_monitoring.py
   python examples/robust_factor_model.py
   python examples/plot_robust_pca_yield_curve.py
   python examples/plot_cellpca_process_spectra.py
   python examples/plot_robust_multilinear_pca.py
   python examples/run_use_case_gallery.py --group pca

``distributionally_robust_pca.py`` saves its figures under
``results/use_cases/distributionally_robust_pca``. Refresh them with:

.. code-block:: bash

   python docs/generate_gallery_assets.py --only distributionally_robust_pca distributionally_robust_pca_drift_monitoring

``robust_factor_model.py`` saves its figures under
``results/use_cases/robust_factor_model``.  Refresh the embedded factor-model
assets with:

.. code-block:: bash

   python docs/generate_gallery_assets.py --only robust_factor_model

Detailed pages
--------------

Open an example from the cards above. Individual examples are intentionally
kept out of the global documentation sidebar.
