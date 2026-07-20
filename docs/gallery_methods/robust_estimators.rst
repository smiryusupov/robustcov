Robust covariance, scatter, and precision estimators
====================================================

Use this section when the primary output is a robust covariance, scatter,
precision matrix, sparse graph, or distance geometry.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/adversarial_covariance_filtering.html">
       <div class="gallery-card-placeholder">Spectral<br>filter</div>
       <h3>Adversarial row filtering</h3>
       <p>Filter structured whole-row attacks in an approximately Gaussian covariance problem.</p>
     </a>
     <a class="gallery-card" href="../gallery/finance_risk.html">
       <img src="../_static/gallery/finance_risk/covariance.png" alt="Robust covariance on heavy-tailed returns">
       <h3>Heavy-tail scatter</h3>
       <p>Compare empirical covariance with regularized Cauchy scatter on return-like data.</p>
     </a>
     <a class="gallery-card" href="../gallery/mrcd_high_dimensional_outliers.html">
       <div class="gallery-card-placeholder">MRCD<br>p near n</div>
       <h3>High-dimensional covariance</h3>
       <p>Use MRCD when the feature count approaches or exceeds the sample size.</p>
     </a>
     <a class="gallery-card" href="../gallery/cellmcd_market_data.html">
       <img src="../_static/gallery/cellmcd_market_data/cell_residual_map.png" alt="CellMCD residual map">
       <h3>Cellwise covariance</h3>
       <p>Retain useful rows while identifying isolated bad cells and missing entries.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_graphical_lasso_market_network.html">
       <img src="../_static/gallery/robust_graphical_lasso_market_network/robust_network.png" alt="Robust sparse precision network">
       <h3>Sparse precision</h3>
       <p>Estimate conditional-dependence networks from robust scatter matrices.</p>
     </a>
     <a class="gallery-card" href="../gallery/spatial_sign_graphical_lasso.html">
       <img src="../_static/gallery/spatial_sign_graphical_lasso/spatial_sign_network.png" alt="Spatial-sign graph">
       <h3>Spatial-sign precision</h3>
       <p>Build scale-free sparse graphs under radial heavy tails.</p>
     </a>
     <a class="gallery-card" href="../geometry.html">
       <div class="gallery-card-placeholder">SPD<br>geometry</div>
       <h3>Scatter geometry</h3>
       <p>Compare and monitor covariance matrices with affine-invariant geometry.</p>
     </a>
   </div>

Runnable examples
-----------------

.. code-block:: bash

   python examples/adversarial_covariance_filtering.py
   python examples/use_case_finance_risk.py
   python examples/plot_mrcd_high_dimensional_outliers.py
   python examples/plot_cellmcd_market_data.py
   python examples/plot_robust_graphical_lasso_market_network.py
   python examples/run_use_case_gallery.py --group robust

Detailed pages
--------------

Open an example from the cards above. Individual examples are intentionally
kept out of the global documentation sidebar.
