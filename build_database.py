import argparse
import sqlite3
from pathlib import Path
import pandas as pd


def load_csv_to_sqlite(csv_path: Path, conn: sqlite3.Connection, table_name: str) -> None:
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def main(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    root = Path(__file__).resolve().parent
    datasets = {
        "master_user_app_usage_categorized.csv": "user_app_usage",
        "sy4836_17576562278687568.csv": "telecom_metrics",
        "user_app_tiers.csv": "user_app_tiers",
    }
    for filename, table in datasets.items():
        load_csv_to_sqlite(root / filename, conn, table)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CSV datasets into a SQLite database")
    parser.add_argument("--db", default="apponomics.db", help="Path to SQLite database file")
    args = parser.parse_args()
    main(args.db)
