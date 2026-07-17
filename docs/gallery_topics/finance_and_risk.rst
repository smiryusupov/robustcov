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
- :doc:`Systemic and idiosyncratic market shocks <../gallery/robust_pca_market_risk>`
- :doc:`Finance risk <../gallery/finance_risk>`
- :doc:`Portfolio stress <../gallery/portfolio_stress>`
