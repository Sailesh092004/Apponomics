"""Evaluation utilities for Apponomics.

This module generates diagnostic plots for marketing spend data, including:
- Distribution of spend-to-cap ratios.
- Churn probability by user tier.
- PCA-based cluster scatter plots.

The functions expect a pandas ``DataFrame`` and rely on matplotlib/seaborn for
visualisation.  A small demonstration dataset is generated when executing the
module as a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class PlotConfig:
    """Configuration for generated plots."""

    style: str = "whitegrid"
    palette: str = "tab10"
    figsize: tuple[int, int] = (18, 5)
    bins: int = 20


def plot_spend_cap_distribution(
    df: pd.DataFrame,
    spend_col: str = "spend",
    cap_col: str = "cap",
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the distribution of the ratio between spend and cap.

    Parameters
    ----------
    df:
        DataFrame containing spend and cap columns.
    spend_col, cap_col:
        Column names for spend and cap values.
    ax:
        Optional existing ``Axes`` to plot on.
    """

    ratio = df[spend_col] / df[cap_col]
    ax = ax or plt.gca()
    sns.histplot(ratio, bins=20, kde=True, ax=ax)
    ax.set_title("Spend/Cap Distribution")
    ax.set_xlabel("Spend / Cap")
    ax.set_ylabel("Frequency")
    return ax


def plot_churn_probability_by_tier(
    df: pd.DataFrame,
    churn_col: str = "churned",
    tier_col: str = "tier",
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot churn probability for each user tier."""

    churn_prob = df.groupby(tier_col)[churn_col].mean().reset_index()
    ax = ax or plt.gca()
    sns.barplot(data=churn_prob, x=tier_col, y=churn_col, ax=ax)
    ax.set_title("Churn Probability by Tier")
    ax.set_ylabel("Churn Probability")
    ax.set_xlabel("Tier")
    return ax


def plot_pca_clusters(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    cluster_col: Optional[str] = None,
    *,
    n_components: int = 2,
    n_clusters: int = 3,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of PCA components coloured by cluster.

    Parameters
    ----------
    df:
        DataFrame containing the feature columns.
    feature_cols:
        Columns to feed into PCA.
    cluster_col:
        Optional column containing pre-computed cluster labels.  If omitted,
        ``KMeans`` clustering is applied on the PCA components.
    n_components:
        Number of principal components to compute.
    n_clusters:
        Number of clusters for ``KMeans`` when ``cluster_col`` is ``None``.
    ax:
        Optional existing ``Axes`` to plot on.
    """

    features = df[list(feature_cols)].dropna()
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(features)

    if cluster_col and cluster_col in df:
        clusters = df.loc[features.index, cluster_col].values
    else:
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        clusters = km.fit_predict(comps)

    ax = ax or plt.gca()
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=clusters, palette="tab10", ax=ax)
    ax.set_title("PCA Cluster Scatterplot")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", loc="best")
    return ax


def generate_demo_data(rows: int = 300) -> pd.DataFrame:
    """Create a synthetic dataset for demonstration purposes."""

    import numpy as np

    rng = np.random.default_rng(0)
    tiers = ["basic", "premium", "enterprise"]
    return pd.DataFrame(
        {
            "spend": rng.gamma(shape=2.0, scale=100, size=rows),
            "cap": rng.gamma(shape=2.0, scale=120, size=rows),
            "tier": rng.choice(tiers, size=rows),
            "churned": rng.integers(0, 2, size=rows),
            "feature1": rng.normal(size=rows),
            "feature2": rng.normal(size=rows),
            "feature3": rng.normal(size=rows),
        }
    )


def main(csv_path: Optional[str] = None) -> str:
    """Generate evaluation plots.

    Parameters
    ----------
    csv_path:
        Optional path to a CSV dataset.  If omitted, synthetic data is used.

    Returns
    -------
    Path to the saved figure.
    """

    df = pd.read_csv(csv_path) if csv_path else generate_demo_data()
    cfg = PlotConfig()
    sns.set(style=cfg.style)
    fig, axes = plt.subplots(1, 3, figsize=cfg.figsize)

    plot_spend_cap_distribution(df, ax=axes[0])
    plot_churn_probability_by_tier(df, ax=axes[1])
    plot_pca_clusters(df, ["feature1", "feature2", "feature3"], ax=axes[2])

    plt.tight_layout()
    out_path = "evaluation_plots.png"
    fig.savefig(out_path)
    return out_path


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else None
    output = main(csv)
    print(f"Saved plots to {output}")
