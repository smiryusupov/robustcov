ICA, SOBI, and source separation
================================

Use this section when the goal is to recover latent signals rather than only a
covariance matrix or principal subspace.

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="../gallery/ica_two_scatter.html">
       <img src="../_static/gallery/ica_two_scatter/source_recovery.png" alt="Robust two-scatter ICA source recovery">
       <h3>Robust two-scatter ICA</h3>
       <p>Separate independent sources with robust whitening and a bounded radial scatter.</p>
     </a>
     <a class="gallery-card" href="../gallery/sobi_source_separation.html">
       <img src="../_static/gallery/sobi_source_separation/source_recovery.png" alt="Robust SOBI source recovery">
       <h3>Robust SOBI</h3>
       <p>Recover temporally correlated sources under heavy tails and impulsive contamination.</p>
     </a>
     <a class="gallery-card" href="../source_separation_factor_models.html">
       <div class="gallery-card-placeholder">BSS<br>guide</div>
       <h3>Method guide</h3>
       <p>Assumptions, fitted attributes, recovery metrics, and backend choices.</p>
     </a>
   </div>

Runnable examples
-----------------

.. code-block:: bash

   python examples/ica_two_scatter.py
   python examples/sobi_source_separation.py
   python examples/run_use_case_gallery.py --group ica

The standalone scripts save plots under ``results/use_cases/ica_two_scatter``
and ``results/use_cases/sobi_source_separation``.  To refresh the copies
embedded in the documentation, run:

.. code-block:: bash

   python docs/generate_gallery_assets.py --only ica_two_scatter sobi_source_separation

Detailed pages
--------------

Open an example from the cards above. Individual examples are intentionally
kept out of the global documentation sidebar.
