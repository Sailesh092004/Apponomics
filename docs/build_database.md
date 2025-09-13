# build_database.py

`build_database.py` loads the CSV datasets in the repository into a SQLite
database. This is useful for experiments that require SQL queries rather than
raw CSV access.

Example usage:

```bash
python build_database.py --db apponomics.db
```

This will create (or overwrite) `apponomics.db` with three tables:
`user_app_usage`, `telecom_metrics`, and `user_app_tiers`.
