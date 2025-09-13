# Apponomics

Utility script for evaluating trained machine learning models.

## evaluate.py

`evaluate.py` loads a serialized model and test dataset, computes task-specific metrics (accuracy, F1, confusion matrix, MAE, RMSE, AUC-ROC, silhouette score) and generates SHAP-based feature importance plots.

Example usage:

```bash
python evaluate.py --model path/to/model.pkl --data path/to/test.csv --task classification --target label
```

The script writes metrics to `evaluation_output/metrics.json` and saves SHAP plots in the same directory.

## build_database.py

`build_database.py` loads the CSV datasets in the repository into a SQLite
database. This is useful for experiments that require SQL queries rather than
raw CSV access.

Example usage:

```bash
python build_database.py --db apponomics.db
```

This will create (or overwrite) `apponomics.db` with three tables:
`user_app_usage`, `telecom_metrics`, and `user_app_tiers`.
