import argparse
import json
import math
import os
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plot styling."""

    palette: str = "tab10"
    bins: int = 10


# Default plot configuration used by plotting functions
CFG = PlotConfig()


def load_model(model_path):
    """Load a trained model from disk.

    Tries to use joblib first and falls back to pickle if joblib is not
    available. This keeps the script importable even in minimal
    environments where optional dependencies are missing.
    """
    try:
        import joblib  # type: ignore
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)


def load_data(data_path, target_col=None):
    """Load test data from a CSV file."""
    import pandas as pd  # Imported lazily to avoid hard dependency during --help
    df = pd.read_csv(data_path)
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X, y = df, None
    return X, y


def evaluate_classification(model, X, y):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="weighted")),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

    # Compute AUC-ROC when probability estimates are available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                metrics["auc_roc"] = float(roc_auc_score(y, proba[:, 1]))
            else:
                metrics["auc_roc"] = float(roc_auc_score(y, proba, multi_class="ovr"))
        except Exception:
            pass
    return metrics


def evaluate_regression(model, X, y):
    """Compute regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = math.sqrt(mean_squared_error(y, preds))
    return {"mae": float(mae), "rmse": float(rmse)}


def evaluate_clustering(model, X):
    """Compute clustering metrics."""
    from sklearn.metrics import silhouette_score

    if hasattr(model, "predict"):
        labels = model.predict(X)
    else:
        labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return {"silhouette_score": float(score)}


def plot_pca_clusters(model, X, features, output_dir, cfg: PlotConfig = CFG):
    """Create a 2D PCA scatter plot of clustered data.

    The function attempts to import the required optional dependencies at
    runtime. If they are not available a message is printed and plotting is
    skipped, allowing the rest of the evaluation to proceed without error.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover - best effort
        print(f"PCA plotting skipped: {exc}")
        return

    data = X[features] if features else X

    if hasattr(model, "predict"):
        labels = model.predict(data)
    else:  # pragma: no cover - models without predict
        labels = model.fit_predict(data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=labels,
        palette=cfg.palette,
        legend=False,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Cluster Plot")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_clusters.png"))
    plt.close()


def plot_spend_cap_distribution(df, output_dir, cfg: PlotConfig = CFG):
    """Plot a histogram of spend cap values."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Spend cap distribution plot skipped: {exc}")
        return

    if "spend_cap" not in df.columns:
        print("Spend cap distribution plot skipped: 'spend_cap' column missing")
        return

    sns.histplot(data=df, x="spend_cap", bins=cfg.bins, palette=cfg.palette)
    plt.xlabel("Spend Cap")
    plt.ylabel("Count")
    plt.title("Spend Cap Distribution")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spend_cap_distribution.png"))
    plt.close()


def plot_tier_counts(df, output_dir, cfg: PlotConfig = CFG):
    """Plot the counts of a tier column if present."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Tier count plot skipped: {exc}")
        return

    tier_col = None
    for candidate in ("tier", "tier_label"):
        if candidate in df.columns:
            tier_col = candidate
            break
    if tier_col is None:
        print("Tier count plot skipped: no tier column found")
        return

    counts = df[tier_col].value_counts().reset_index()
    counts.columns = [tier_col, "count"]
    sns.barplot(data=counts, x=tier_col, y="count", palette=cfg.palette)
    plt.xlabel(tier_col.replace("_", " ").title())
    plt.ylabel("Count")
    plt.title("Tier Counts")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tier_counts.png"))
    plt.close()


def create_shap_plots(model, X, output_dir):
    """Generate SHAP-based feature importance plots.

    If SHAP or matplotlib are not installed the function will print a
    warning and continue without raising an exception.
    """
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        os.makedirs(output_dir, exist_ok=True)

        # Summary beeswarm plot
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary.png"))
        plt.close()

        # Bar plot of mean absolute SHAP values
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_bar.png"))
        plt.close()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"SHAP analysis skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
    parser.add_argument("--model", required=True, help="Path to a serialized model (joblib or pickle)")
    parser.add_argument("--data", required=True, help="Path to CSV file containing test data")
    parser.add_argument("--task", required=True, choices=["classification", "regression", "clustering"],
                        help="Type of machine learning task")
    parser.add_argument("--target", help="Target column name for supervised tasks")
    parser.add_argument("--output", default="evaluation_output", help="Directory to save metrics and plots")
    parser.add_argument("--features", nargs="+",
                        help="Feature columns to use. If omitted numeric columns are inferred")
    args = parser.parse_args()

    model = load_model(args.model)
    X, y = load_data(args.data, args.target)

    if args.features:
        features = args.features
    else:
        features = X.select_dtypes(include="number").columns.tolist()

    if args.task in ("classification", "regression") and y is None:
        raise ValueError("Target column must be provided for supervised tasks")

    data_for_eval = X[features] if args.task == "clustering" else X

    if args.task == "classification":
        metrics = evaluate_classification(model, data_for_eval, y)
    elif args.task == "regression":
        metrics = evaluate_regression(model, data_for_eval, y)
    else:
        metrics = evaluate_clustering(model, data_for_eval)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP plots for feature importances
    create_shap_plots(model, data_for_eval, args.output)

    # Additional exploratory plots honoring the plot configuration
    plot_spend_cap_distribution(X, args.output, CFG)
    plot_tier_counts(X, args.output, CFG)

    if args.task == "clustering":
        plot_pca_clusters(model, data_for_eval, features, args.output, CFG)

    # Print metrics to stdout for convenience
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
