Kaggle and external examples
============================

Kaggle notebooks are a practical way to make robust covariance useful to a wider ML audience.  Users usually arrive with a problem such as fraud detection, predictive maintenance, finance stress detection, or medical screening; they do not usually start by searching for Tyler or Cauchy scatter estimators.

The examples in ``examples_external/`` are therefore designed as optional, publishable templates.  They are **not part of tests** and are not required for the core docs build because they need external downloads and dataset-specific licenses.

How to use these examples
-------------------------

1. Download the dataset manually from Kaggle or the dataset provider.
2. Run the matching script with ``--data /path/to/file.csv``.
3. Inspect the generated ``metrics.csv``, plots, and ``summary.md``.
4. Turn the result into a Kaggle notebook if it is competitive or interpretable.

.. code-block:: bash

   python examples_external/kaggle_credit_card_fraud.py \
     --data /path/to/creditcard.csv \
     --outdir results/external/credit_card_fraud

Each script writes:

* ``metrics.csv`` with precision, recall, F1, ROC AUC, PR AUC, and runtime;
* one or more metric plots;
* a robust-score profile plot;
* ``summary.md`` for easy notebook/report copying.

Recommended external targets
----------------------------

.. raw:: html

   <div class="gallery-grid">
     <a class="gallery-card" href="#credit-card-fraud-detection">
       <div class="gallery-card-placeholder">Fraud<br>PR-AUC</div>
       <h3>Credit-card fraud</h3>
       <p>Classic imbalanced anomaly detection.  Good first Kaggle notebook target.</p>
     </a>
     <a class="gallery-card" href="#ieee-cis-transaction-fraud">
       <div class="gallery-card-placeholder">Large<br>tabular</div>
       <h3>IEEE-CIS fraud</h3>
       <p>Use robustcov as a transaction screening score and as a feature for supervised models.</p>
     </a>
     <a class="gallery-card" href="#predictive-maintenance">
       <div class="gallery-card-placeholder">Sensors<br>faults</div>
       <h3>Predictive maintenance</h3>
       <p>Equipment faults often appear as multivariate deviations from normal operation.</p>
     </a>
     <a class="gallery-card" href="#medical-tabular-screening">
       <div class="gallery-card-placeholder">Medical<br>screening</div>
       <h3>Medical tabular screening</h3>
       <p>Robust scores are interpretable patient-level screening features, not clinical decisions.</p>
     </a>
   </div>

Credit-card fraud detection
---------------------------

**Script:** ``examples_external/kaggle_credit_card_fraud.py``

**Expected data:** a CSV such as Kaggle's credit-card fraud dataset with a binary ``Class`` column.

**Why it fits robustcov:** fraud is rare, the feature distribution is non-Gaussian, and robust distances provide an interpretable score for ranking suspicious transactions.

**Recommended metric:** PR AUC first, then F1 at the chosen contamination threshold.  ROC AUC can look deceptively high on very imbalanced fraud data.

.. code-block:: bash

   python examples_external/kaggle_credit_card_fraud.py \
     --data /path/to/creditcard.csv

IEEE-CIS transaction fraud
--------------------------

**Script:** ``examples_external/kaggle_ieee_cis_fraud.py``

**Expected data:** ``train_transaction.csv`` with ``isFraud`` and ``TransactionID`` columns.

**Why it fits robustcov:** robustcov is unlikely to replace a full supervised gradient-boosting competition pipeline, but it can provide fast unsupervised transaction scores, preprocessing filters, and interpretable anomaly profiles.

.. code-block:: bash

   python examples_external/kaggle_ieee_cis_fraud.py \
     --data /path/to/train_transaction.csv \
     --max-rows 100000

Predictive maintenance
----------------------

**Script:** ``examples_external/kaggle_predictive_maintenance.py``

**Expected data:** a sensor or equipment table with a failure indicator such as ``Machine failure``, ``failure``, or ``target``.

**Why it fits robustcov:** faults often manifest as combinations of unusual sensor readings rather than one extreme variable.  Robust covariance gives an interpretable multivariate score.

.. code-block:: bash

   python examples_external/kaggle_predictive_maintenance.py \
     --data /path/to/predictive_maintenance.csv

Medical tabular screening
-------------------------

**Script:** ``examples_external/kaggle_medical_screening.py``

**Expected data:** a diagnostic feature table with a label such as ``diagnosis``, ``target``, ``Class``, or ``outcome``.

**Why it fits robustcov:** robust scores can support exploratory screening and data-quality analysis.  They should not be interpreted as clinical decisions.

.. code-block:: bash

   python examples_external/kaggle_medical_screening.py \
     --data /path/to/medical.csv \
     --label-column diagnosis \
     --positive-label malignant


Finance market stress and rolling regimes
-----------------------------------------

**Scripts:** ``examples_external/finance_market_stress.py`` and
``examples_external/finance_rolling_window_anomaly.py``

**Expected data:** price or return CSV with a date column and one column per
asset.  This can come from Kaggle, Yahoo Finance exports, Bloomberg, Quandl, or
another provider.

.. code-block:: bash

   python examples_external/finance_market_stress.py \
     --prices /path/to/prices.csv \
     --outdir results/external/finance_market_stress

   python examples_external/finance_rolling_window_anomaly.py \
     --prices /path/to/prices.csv \
     --window 20 \
     --step 5 \
     --outdir results/external/finance_rolling_window

Collecting external results
---------------------------

After running several external examples, collect their ``metrics.csv`` files into
a compact registry:

.. code-block:: bash

   python examples_external/collect_external_results.py \
     --root results/external \
     --outdir results/external_registry

This writes ``external_results.csv``, ``external_results.md``, and
``external_results.html``.

Notebook template
-----------------

A copyable Kaggle notebook template is included at:

.. code-block:: text

   examples_external/notebooks/robustcov_kaggle_template.ipynb

Use the scripts when you want reproducible local runs.  Use the notebook template when publishing a short Kaggle walkthrough.

What makes a good Kaggle notebook?
----------------------------------

A useful robustcov notebook should be short and evidence-driven:

* load data and define the target;
* compare robustcov with IsolationForest, LOF, and other familiar baselines;
* report PR AUC, F1, ROC AUC, and runtime;
* show score distributions or robust-distance profiles;
* explain where robustcov is competitive and where supervised models are still better.

