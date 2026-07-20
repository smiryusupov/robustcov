# Optional Kaggle / external examples

These examples are not run during the normal Sphinx build. Large/raw datasets are never committed to the repository. Dataset-loader tests use tiny local archives and remain fully offline; real-data workflows run only after an explicit download or a manually supplied archive.

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
| `gas_sensor_drift_dro_pca.py` | UCI gas-sensor temporal batches | exploratory batch-level drift analysis; not a reviewed snapshot |
| `cmapss_dro_pca_monitoring.py` | NASA C-MAPSS FD001--FD004 | operating-regime-aware degradation monitoring |


## Cached external datasets

```bash
export ROBUSTCOV_DATA_DIR="$HOME/data/robustcov"
python -m robustcov.datasets list
python -m robustcov.datasets fetch gas_sensor_drift
python -m robustcov.datasets fetch cmapss --subset FD002
```

Downloads are explicit, atomic, checksum/fingerprint validated, and safely extracted. A manually downloaded archive can be supplied with `--archive` to either DRO-PCA script. Raw archives and extracted files remain under the user cache; only result CSVs and figures are written under `results/external/`.

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

## Publish reviewed documentation snapshots

Real-data protocols write a complete local workspace under `results/external/`.
After reviewing a run, publish only the approved aggregate figures and summary
table into the documentation:

```bash
python scripts/publish_external_snapshot.py publish cmapss_fd002 \
  --results results/external/cmapss_fd002 \
  --command "python examples_external/cmapss_dro_pca_monitoring.py --download --subset FD002 --outdir results/external/cmapss_fd002"

python scripts/publish_external_snapshot.py check
```

The publisher currently accepts only the reviewed C-MAPSS FD002 and FD004
profiles. It excludes `window_scores.csv`, local cache paths, raw data, and
embeddings, records file digests, and creates the Sphinx result page and gallery
card. The gas-sensor script remains exploratory and is not published as a
reference snapshot. See `docs/external_snapshot_policy.rst`.

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
