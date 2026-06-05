Kaggle and external dataset roadmap
===================================

The built-in gallery intentionally uses sklearn datasets and synthetic generators so that examples
run without network access, API keys, or large downloads.  The next expansion path is to add optional
Kaggle/external-data examples behind explicit commands.

Why optional?
-------------

External datasets are useful for adoption, but they should not make tests or documentation builds
fragile.  Large downloads, license terms, credentials, and changing dataset URLs should stay outside
core package validation.

Candidate external examples
---------------------------

.. list-table:: Kaggle-ready use-case candidates
   :header-rows: 1

   * - Application
     - Dataset family
     - Why it fits robustcov
   * - Credit-card fraud
     - Kaggle Credit Card Fraud Detection
     - Extreme class imbalance and tabular anomaly screening are natural robust-distance use cases.
   * - IEEE-CIS transaction fraud
     - IEEE-CIS Fraud Detection competition data
     - High-dimensional transaction features make robust preprocessing and screening useful.
   * - Industrial equipment monitoring
     - Predictive-maintenance / sensor fault datasets
     - Robust distance profiles are easy for engineers to inspect; compare honestly against IsolationForest.
   * - Medical tabular screening
     - UCI/Kaggle diagnostic datasets
     - Robust distances provide interpretable patient-level anomaly scores.

Suggested implementation pattern
--------------------------------

External examples should follow this pattern:

.. code-block:: bash

   python examples_external/kaggle_credit_card_fraud.py --data /path/to/creditcard.csv

The script should never download data silently.  It should print the expected schema, save metrics,
create plots, and write a short Markdown summary into ``results/external/``.

Documentation strategy
----------------------

Each external dataset should get a page with:

* where to obtain the data;
* expected file names and columns;
* preprocessing notes;
* baseline comparisons;
* robustcov output and plots;
* limitations and licensing caveats.
