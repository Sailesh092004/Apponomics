# Apponomics

Utility scripts for working with mobile app usage data. The project now
provides a basic data generation and model training pipeline alongside a
functional evaluation script and Streamlit demo app.

## Scripts

- `scripts/generate_data.py` – create synthetic datasets for experimentation.
- `scripts/train.py` – train classification, regression or clustering models.
- `scripts/evaluate.py` – evaluate a serialized model on a dataset and output
  metrics, SHAP plots, and optional clustering visuals.

Example usage for data generation and training:

```bash
python scripts/generate_data.py --rows 1000 --output data.csv
python scripts/train.py --data data.csv --task classification --target churn --model model.pkl
```

Example usage for the evaluation script:

```bash
python scripts/evaluate.py --model path/to/model.pkl --data path/to/test.csv \
    --task classification --target label
```

For clustering tasks you can optionally specify which feature columns to use
for evaluation and PCA cluster visualisation. If omitted, all numeric columns
are used automatically:

```bash
python scripts/evaluate.py --model path/to/model.pkl --data path/to/test.csv \
    --task clustering --features spend sessions
```

The script writes metrics to `evaluation_output/metrics.json` and saves SHAP
plots (and clustering plots when applicable) in the same directory.

## app.py

`app.py` provides a simple [Streamlit](https://streamlit.io/) interface for
uploading a CSV file and viewing model predictions.

## build_database.py

`build_database.py` loads the CSV datasets in the repository into a SQLite database. This is useful for experiments that require SQL queries rather
than raw CSV access.

Example usage:

```bash
python build_database.py --db apponomics.db
```

This will create (or overwrite) `apponomics.db` with three tables:
`user_app_usage`, `telecom_metrics`, and `user_app_tiers`.
