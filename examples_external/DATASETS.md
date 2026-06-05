# Optional external/Kaggle datasets

These datasets are **not** part of the robustcov test suite. Download them manually
according to their licenses and run the matching script.

| Use case | Suggested dataset/source | Script |
|---|---|---|
| Credit-card fraud | Kaggle creditcard fraud CSV with `Class` column | `kaggle_credit_card_fraud.py` |
| IEEE-CIS fraud | Kaggle IEEE-CIS train transaction CSV | `kaggle_ieee_cis_fraud.py` |
| Network intrusion | UNSW-NB15 / NSL-KDD style CSV | `kaggle_network_intrusion.py` |
| Predictive maintenance | Kaggle predictive maintenance / machine failure CSV | `kaggle_predictive_maintenance.py` |
| Medical screening | medical tabular CSV with diagnosis/outcome label | `kaggle_medical_screening.py` |
| Market stress | asset price or return CSV, date + one column per asset | `finance_market_stress.py` |
| Rolling market regimes | asset price or return CSV | `finance_rolling_window_anomaly.py` |

Recommended result workflow:

```bash
python examples_external/<script>.py --data /path/to/data.csv --outdir results/external/<name>
python examples_external/collect_external_results.py --root results/external --outdir results/external_registry
```

## No-download finance data

The finance examples can be run without an external dataset:

```bash
python examples_external/make_synthetic_prices.py --out examples_external/data/prices.csv
```

The generated CSV has a `date` column and one price column per synthetic asset.
It contains correlated heavy-tailed returns plus injected stress windows, so it
is useful for testing the market-stress and rolling-window scripts before moving
to Kaggle, Yahoo Finance, Bloomberg, or internal market data.
