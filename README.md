# Apponomics

Utilities for analysing marketing data.  The `evaluate.py` script produces a
combined figure containing:

1. **Spend-cap distribution** – histogram of spend-to-cap ratios.
2. **Churn probability by tier** – bar chart of churn rates for each customer tier.
3. **PCA-based cluster scatterplot** – visualises clusters in the first two
   principal components.

Run the script directly to generate the plots using a synthetic dataset:

```bash
python evaluate.py
```

Provide a CSV file as an argument to analyse your own dataset:

```bash
python evaluate.py path/to/data.csv
```
