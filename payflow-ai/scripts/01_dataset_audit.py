from pathlib import Path
import json

import mlflow
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"

DATA_PATH = Path(
    "data/raw/upi_transactions_2024.csv"
)

EXPERIMENT_NAME = "payflow-data-audit"

def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    df = dataframe.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    return df

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)

df = normalize_columns(df)
print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nTransaction status:")
print(
    df["transaction_status"]
    .value_counts(dropna=False)
)

print("\nFraud flag:")
print(
    df["fraud_flag"]
    .value_counts(dropna=False)
)

required_columns = {
    "transaction_id",
    "timestamp",
    "transaction_type",
    "merchant_category",
    "amount_inr",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "fraud_flag",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
}

missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )
if not df["amount_inr"].ge(0).all():
    raise ValueError(
        "Negative transaction amount found."
    )


if not df["hour_of_day"].between(
    0,
    23,
).all():
    raise ValueError(
        "Invalid hour_of_day found."
    )


valid_statuses = {
    "SUCCESS",
    "FAILED",
}


actual_statuses = set(
    df["transaction_status"]
    .dropna()
    .unique()
)


if not actual_statuses.issubset(
    valid_statuses
):
    raise ValueError(
        f"Unexpected transaction status: "
        f"{actual_statuses}"
    )

dataset = mlflow.data.from_pandas(
    df,
    source=str(DATA_PATH),
    name="upi-transactions-2024",
    targets="transaction_status",
)

row_count = len(df)

column_count = len(df.columns)

missing_cells = int(
    df.isna().sum().sum()
)

duplicate_transactions = int(
    df["transaction_id"]
    .duplicated()
    .sum()
)

success_rate = float(
    df["transaction_status"]
    .eq("SUCCESS")
    .mean()
)

fraud_rate = float(
    pd.to_numeric(
        df["fraud_flag"],
        errors="coerce",
    )
    .fillna(0)
    .mean()
)

profile = {
    "rows": row_count,
    "columns": column_count,
    "missing_cells": missing_cells,
    "duplicate_transactions":
        duplicate_transactions,
    "success_rate":
        success_rate,
    "fraud_rate":
        fraud_rate,
    "columns_list":
        df.columns.tolist(),
}
profile_path = Path(
    "artifacts/dataset_profile.json"
)

profile_path.write_text(
    json.dumps(
        profile,
        indent=2,
    ),
    encoding="utf-8",
)
# Start an MLflow run to log the dataset audit results
with mlflow.start_run(
    run_name="raw-dataset-audit"
):

    mlflow.log_input(
        dataset,
        context="raw",
    )

    mlflow.log_params(
        {
            "project":
                "payflow-ai",

            "dataset":
                "upi-transactions-2024",

            "currency":
                "INR",

            "target":
                "transaction_status",

            "stage":
                "raw-data-audit",
        }
    )

    mlflow.log_metrics(
        {
            "row_count":
                row_count,

            "column_count":
                column_count,

            "missing_cells":
                missing_cells,

            "duplicate_transactions":
                duplicate_transactions,

            "success_rate":
                success_rate,

            "fraud_rate":
                fraud_rate,
        }
    )

    mlflow.log_artifact(
        str(profile_path),
        artifact_path="data_quality",
    )