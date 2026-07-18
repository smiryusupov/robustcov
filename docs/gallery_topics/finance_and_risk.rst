Finance and risk
================

Robust covariance and robust principal subspaces are useful in finance because
returns and curve changes are heavy-tailed, correlated, and sensitive to stress
regimes.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/robust_pca_yield_curve.html">
       <img src="../_static/gallery/robust_pca_yield_curve/factor_loadings.png" alt="Robust PCA yield curve factors">
       <h3>Yield-curve factors</h3>
       <p>Recover stable level, slope, and curvature factors and separate familiar shocks from quote dislocations.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_pca_subspace_stability.html">
       <img src="../_static/gallery/robust_pca_subspace_stability/principal_angle_distribution.png" alt="Bootstrap stability of robust yield curve factors">
       <h3>Factor stability</h3>
       <p>Bootstrap loadings, eigenvalues, and principal angles before treating PCA factors as repeatable.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_pca_dependent_stability.html">
       <img src="../_static/gallery/robust_pca_dependent_stability/principal_angle_distribution.png" alt="Stationary bootstrap uncertainty for serially dependent robust PCA factors">
       <h3>Dependent factor stability</h3>
       <p>Use stationary blocks instead of independent rows when factor scores are serially dependent.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_pca_market_risk.html">
       <img src="../_static/gallery/robust_pca_market_risk/outlier_map.png" alt="Robust PCA market shock outlier map">
       <h3>Systemic versus idiosyncratic shocks</h3>
       <p>Use score and orthogonal distances to distinguish common-factor stress from isolated asset moves.</p>
     </a>
     <a class="gallery-card" href="../gallery/finance_risk.html">
       <img src="../_static/gallery/finance_risk/covariance.png" alt="Finance risk covariance">
       <h3>Finance risk covariance</h3>
       <p>Small-sample heavy-tail covariance for multi-asset return features.</p>
     </a>
     <a class="gallery-card" href="../gallery/portfolio_stress.html">
       <img src="../_static/gallery/portfolio_stress/covariance.png" alt="Portfolio stress monitoring">
       <h3>Portfolio stress monitoring</h3>
       <p>Compare empirical and robust risk estimates under stress contamination.</p>
     </a>
     <a class="gallery-card" href="../gallery/cellmcd_market_data.html">
       <img src="../_static/gallery/cellmcd_market_data/cell_residual_map.png" alt="CellMCD market-data residual map">
       <h3>Isolated bad ticks</h3>
       <p>Flag individual corrupted returns, impute missing quotes, and retain the clean cells in each day.</p>
     </a>
     <a class="gallery-card" href="../gallery/robust_graphical_lasso_market_network.html">
       <img src="../_static/gallery/robust_graphical_lasso_market_network/robust_network.png" alt="Robust sparse market network">
       <h3>Sparse market network</h3>
       <p>Estimate conditional asset links from a cellwise-robust scatter matrix.</p>
     </a>
   </div>

How to use this topic
---------------------

Start with the yield-curve page for an interpretable robust PCA factor example.
Use the cross-asset page when the operational question is whether a market event
is systemic or idiosyncratic.  The covariance pages focus directly on robust
risk matrices rather than dimension reduction.

Detailed pages
--------------

- :doc:`Robust yield-curve factors <../gallery/robust_pca_yield_curve>`
- :doc:`Bootstrap factor stability <../gallery/robust_pca_subspace_stability>`
- :doc:`Dependent bootstrap factor stability <../gallery/robust_pca_dependent_stability>`
- :doc:`Systemic and idiosyncratic market shocks <../gallery/robust_pca_market_risk>`
- :doc:`Finance risk <../gallery/finance_risk>`
- :doc:`Portfolio stress <../gallery/portfolio_stress>`
- :doc:`Cellwise market-data cleaning <../gallery/cellmcd_market_data>`
- :doc:`Sparse market network <../gallery/robust_graphical_lasso_market_network>`
