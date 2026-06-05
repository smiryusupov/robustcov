External and Kaggle gallery
===========================

This page is the single entry point for optional Kaggle and external-data
examples.  These examples are **not part of tests** because they require manual
downloads, dataset-specific licenses, or larger local files.

The goal is not to claim that ``robustcov`` wins everywhere.  The goal is to
show where robust covariance gives a strong advantage, where it is competitive,
where it is mainly diagnostic, and where another method is better.

How to read the cards
---------------------

.. list-table:: Result labels
   :header-rows: 1

   * - Label
     - Meaning
   * - Strong win
     - robustcov clearly improves the most relevant metric against common unsupervised baselines.
   * - Competitive
     - robustcov is close to the best method, or wins one metric but loses another.
   * - Competitive, slow
     - robustcov improves quality but runtime is currently a weakness.
   * - Not best
     - another baseline performs better; the robustcov result is still reported for transparency.
   * - Diagnostic
     - there are no ground-truth labels, but robust distances provide interpretable stress/anomaly rankings.

Recommended result pages
------------------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="external_results/credit_card_fraud.html">
       <img src="_static/external_results/credit_card_fraud/pr_auc.png" alt="Credit-card fraud PR-AUC comparison">
       <h3>Credit-card fraud</h3>
       <p><strong>Strong win.</strong> FastMCD PR-AUC 0.712 and F1 0.801 on a classic imbalanced fraud dataset.</p>
     </a>
     <a class="gallery-card" href="external_results/predictive_maintenance.html">
       <img src="_static/external_results/predictive_maintenance/f1.png" alt="Predictive maintenance F1 comparison">
       <h3>Predictive maintenance</h3>
       <p><strong>Competitive.</strong> robustcov gives the best F1, while IsolationForest has stronger PR-AUC and speed.</p>
     </a>
     <a class="gallery-card" href="external_results/finance_market_stress.html">
       <img src="_static/external_results/finance_market_stress/top_stress_days.png" alt="Top finance stress days">
       <h3>Finance market stress</h3>
       <p><strong>Diagnostic.</strong> RegularizedCauchy ranks unusual cross-asset return days.</p>
     </a>
     <a class="gallery-card" href="external_results/finance_rolling_window.html">
       <img src="_static/external_results/finance_rolling_window/top_stress_windows.png" alt="Top anomalous rolling finance windows">
       <h3>Rolling market regimes</h3>
       <p><strong>Diagnostic.</strong> Window-level features identify abnormal volatility/correlation regimes.</p>
     </a>
   </div>

Honest secondary results
------------------------

.. raw:: html

   <div class="gallery-grid compact-gallery-grid">
     <a class="gallery-card" href="external_results/ieee_cis_fraud.html">
       <img src="_static/external_results/ieee_cis_fraud/runtime.png" alt="IEEE-CIS runtime comparison">
       <h3>IEEE-CIS fraud</h3>
       <p><strong>Competitive, slow.</strong> Best tested unsupervised quality, but runtime is a major weakness.</p>
     </a>
     <a class="gallery-card" href="external_results/medical_screening.html">
       <img src="_static/external_results/medical_screening/f1.png" alt="Medical screening F1 comparison">
       <h3>Medical screening</h3>
       <p><strong>Not best.</strong> Useful diagnostic result; EllipticEnvelope wins this benchmark.</p>
     </a>
   </div>

Current documented external results
-----------------------------------

.. list-table:: External result registry
   :header-rows: 1

   * - Dataset / example
     - Status
     - Main method
     - Headline result
     - Notes
   * - Credit-card fraud
     - Strong win
     - FastMCD
     - PR-AUC 0.712, F1 0.801
     - Large metric gap vs common sklearn anomaly baselines.
   * - Predictive maintenance
     - Competitive
     - Auto(StudentTScatter)
     - F1 0.947 vs IsolationForest 0.944
     - IsolationForest is faster and has better PR-AUC.
   * - IEEE-CIS fraud
     - Competitive, slow
     - RegularizedCauchy
     - PR-AUC 0.093 vs IsolationForest 0.084
     - Best tested unsupervised quality, but much slower.
   * - Medical screening
     - Not best
     - Auto(StudentTScatter)
     - PR-AUC 0.567 vs EllipticEnvelope 0.629
     - Honest negative/diagnostic result.
   * - Finance market stress
     - Diagnostic
     - RegularizedCauchy
     - 23 / 899 days detected
     - Top days cluster around stress-like periods.
   * - Rolling-window finance
     - Diagnostic
     - RegularizedCauchy
     - 5 / 176 windows detected
     - Top windows cluster around September stress regimes.

Why UNSW-NB15 is not highlighted
--------------------------------

The commonly used UNSW-NB15 training split can contain a very high attack
fraction.  That makes it less like rare-anomaly detection and more like
unsupervised or semi-supervised classification.  ``robustcov`` may still be
useful there as a risk-ranking diagnostic, but it is not a clean headline
anomaly benchmark for this package.  We therefore do not highlight it in the
external gallery.

Run external examples
---------------------

External examples are optional and dataset-dependent.  The recommended path is:

.. code-block:: bash

   python examples_external/<script>.py --data path/to/data.csv --outdir results/external/<name>
   python examples_external/collect_external_results.py \
     --root results/external \
     --outdir results/external_registry

The scripts, dataset notes, and notebook templates live under ``examples_external/``.
They are intentionally outside the core test suite because Kaggle datasets have
separate licenses, download steps, and file sizes.

.. toctree::
   :maxdepth: 1
   :hidden:

   external_results/credit_card_fraud
   external_results/predictive_maintenance
   external_results/ieee_cis_fraud
   external_results/medical_screening
   external_results/finance_market_stress
   external_results/finance_rolling_window
