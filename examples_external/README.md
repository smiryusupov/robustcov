# Optional Kaggle / external examples

These examples are intentionally **not part of tests** and are not run during the normal Sphinx build. They require datasets that users download manually from Kaggle or another data source.

Why keep them optional?

- Kaggle datasets may require account credentials and acceptance of competition/dataset terms.
- Dataset URLs, schemas, and licenses can change.
- Large downloads make CI and package docs fragile.

## Available scripts

| Script | Dataset family | Main target |
|---|---|---|
| `kaggle_credit_card_fraud.py` | Credit Card Fraud Detection | PR-AUC / F1 fraud screening |
| `kaggle_ieee_cis_fraud.py` | IEEE-CIS Fraud Detection | transaction anomaly scores |
| `kaggle_predictive_maintenance.py` | predictive-maintenance sensor tables | failure screening |
| `kaggle_medical_screening.py` | medical tabular diagnosis datasets | patient-level screening |

## Example

```bash
python examples_external/kaggle_credit_card_fraud.py \
  --data /path/to/creditcard.csv \
  --outdir results/external/credit_card_fraud
```

Each script writes:

- `metrics.csv`
- metric plots such as `pr_auc.png`, `roc_auc.png`, or `f1.png`
- `robust_score_profile.png`
- `summary.md`

## Notebook template

A copyable notebook template is available at:

```text
examples_external/notebooks/robustcov_kaggle_template.ipynb
```

Use the scripts for reproducible local runs and the notebook template for Kaggle publishing.

## Reproducible finance demo without downloads

To run an external-style demo without Kaggle or market data, use:

```bash
python examples_external/run_external_demo_suite.py --synthetic
```

This command generates `examples_external/data/prices.csv`, runs the finance market-stress and rolling-window examples, and writes a compact registry under `results/external_registry/`.

To run the steps manually:

```bash
python examples_external/make_synthetic_prices.py --out examples_external/data/prices.csv
python examples_external/finance_market_stress.py --prices examples_external/data/prices.csv --outdir results/external/finance_market_stress
python examples_external/finance_rolling_window_anomaly.py --prices examples_external/data/prices.csv --outdir results/external/finance_rolling_window
python examples_external/collect_external_results.py --root results/external --outdir results/external_registry
```

Note: network-intrusion datasets with high attack fractions are not highlighted as rare-anomaly benchmarks; use them only as optional diagnostic/risk-ranking experiments.
