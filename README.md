# Apponomics

Apponomics is a demo project showcasing an end‑to‑end machine learning workflow for analysing mobile app performance data. It includes utilities for generating synthetic datasets, training predictive models, evaluating results, and visualising insights through an interactive Streamlit dashboard.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Generation

Generate synthetic training data:
```bash
python scripts/generate_data.py --output data/raw
```

## Model Training

Train a model using the generated data:
```bash
python scripts/train.py --data data/raw --model-dir models/
```

## Evaluation

Evaluate the trained model:
```bash
python scripts/evaluate.py --data data/raw --model models/model.pkl --metrics reports/metrics.json
```

## Streamlit App

Run the Streamlit dashboard to explore predictions and metrics:
```bash
streamlit run app.py
```
