from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd

from dotenv import load_dotenv


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "payflow-payment-success",
)


mlflow.set_tracking_uri(
    TRACKING_URI
)


# =========================================================
# 2. SEARCH RUNS
#
# MLflow returns runs as a Pandas DataFrame.
#
# We order by FAILED F1 because failure detection is one of
# our important signals.
#
# This is NOT yet an automatic production-promotion rule.
# =========================================================

runs = mlflow.search_runs(
    experiment_names=[
        EXPERIMENT_NAME
    ],
    filter_string=(
        "tags.problem = 'payment-success'"
    ),
    order_by=[
        "metrics.val_failed_f1 DESC"
    ],
)


if runs.empty:
    raise RuntimeError(
        "No PayFlow payment-success runs found."
    )


# =========================================================
# 3. SELECT COMPARISON COLUMNS
# =========================================================

wanted_columns = {

    "tags.mlflow.runName":
        "Run Name",

    "params.model_type":
        "Model",

    "metrics.val_accuracy":
        "Accuracy",

    "metrics.val_roc_auc":
        "ROC-AUC",

    "metrics.val_failed_pr_auc":
        "Failed PR-AUC",

    "metrics.val_failed_precision":
        "Failed Precision",

    "metrics.val_failed_recall":
        "Failed Recall",

    "metrics.val_failed_f1":
        "Failed F1",

    "metrics.val_log_loss":
        "Log Loss",

    "metrics.train_seconds":
        "Train Seconds",

    "metrics.predict_ms_per_row":
        "Predict ms/row",

    "run_id":
        "Run ID",
}


# Some older runs may not contain every new metric.
# Keep only columns actually present in MLflow.
available_columns = {
    source: display
    for (
        source,
        display,
    )
    in wanted_columns.items()
    if source in runs.columns
}


comparison = (
    runs[
        list(
            available_columns.keys()
        )
    ]
    .rename(
        columns=available_columns
    )
    .copy()
)


# =========================================================
# 4. PRINT COMPARISON
# =========================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)


print()
print(
    "PayFlow AI — MLflow Run Comparison"
)

print(
    "================================="
)

print()

print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# 5. SAVE COMPARISON AS CSV
# =========================================================

output_path = Path(
    "artifacts/run_comparison.csv"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


comparison.to_csv(
    output_path,
    index=False,
)


print()
print(
    f"Saved comparison: {output_path}"
)