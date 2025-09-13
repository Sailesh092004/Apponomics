# Apponomics

Utility script for evaluating trained machine learning models.

## evaluate.py

`evaluate.py` loads a serialized model and test dataset, computes task-specific metrics (accuracy, F1, confusion matrix, MAE, RMSE, AUC-ROC, silhouette score) and generates SHAP-based feature importance plots.

Example usage:

```bash
python evaluate.py --model path/to/model.pkl --data path/to/test.csv --task classification --target label
```

The script writes metrics to `evaluation_output/metrics.json` and saves SHAP plots in the same directory.
