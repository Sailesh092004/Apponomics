# evaluate.py

`evaluate.py` loads a serialized model and test dataset, computes task-specific metrics (accuracy, F1, confusion matrix, MAE, RMSE, AUC-ROC, silhouette score) and generates SHAP-based feature importance plots.

Example usage for classification:

```bash
python evaluate.py --model path/to/model.pkl --data path/to/test.csv --task classification --target label
```

For clustering tasks you can optionally specify which feature columns to use
for evaluation and PCA cluster visualisation. If omitted, all numeric columns
are used automatically.

```bash
python evaluate.py --model path/to/model.pkl --data path/to/test.csv --task clustering --features spend sessions
```

The script writes metrics to `evaluation_output/metrics.json` and saves SHAP
plots (and clustering plots when applicable) in the same directory.
