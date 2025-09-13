"""Generate a synthetic dataset for training models.

The script produces a CSV file with user spend, session counts,
application tier and churn labels.  It is intentionally simple but
sufficient for demonstrating the training and evaluation pipeline.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> pd.DataFrame:
    """Create a random dataset.

    Parameters
    ----------
    rows: int
        Number of samples to generate.
    seed: int
        Seed for the pseudo-random number generator.
    """

    rng = np.random.default_rng(seed)

    spend = rng.gamma(shape=2.0, scale=50.0, size=rows)
    sessions = rng.poisson(lam=5, size=rows)

    tier = np.where(
        spend >= 500,
        "premium",
        np.where(spend >= 100, "standard", "free"),
    )

    churn_logit = 2.0 - 0.003 * spend - 0.3 * sessions
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
    churn = rng.binomial(1, np.clip(churn_prob, 0, 1))

    df = pd.DataFrame(
        {
            "spend": spend.round(2),
            "sessions": sessions,
            "tier": tier,
            "churn": churn,
        }
    )
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="synthetic_data.csv", help="Output CSV file")
    args = parser.parse_args(argv)

    df = generate(args.rows, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
