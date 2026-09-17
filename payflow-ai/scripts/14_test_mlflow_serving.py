from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from dotenv import load_dotenv

from payflow.data.load import (
    load_transactions,
)

from payflow.features.payment_success import (
    FEATURE_COLUMNS,
)


# =========================================================
# 1. LOAD CONFIGURATION
# =========================================================

load_dotenv()


DATA_PATH = Path(
    os.getenv(
        "PAYFLOW_DATA_PATH",
        "data/raw/upi_transactions_2024.csv",
    )
)


MLFLOW_SERVING_URL = os.getenv(
    "PAYFLOW_MLFLOW_SERVING_URL",
    "http://127.0.0.1:5001",
)


# =========================================================
# 2. LOAD ONE VALID TRANSACTION
#
# We use our actual dataset so:
#
# - categories are valid
# - column names are correct
# - data types closely match training
# =========================================================

df = load_transactions(
    DATA_PATH
)


sample = (
    df[
        FEATURE_COLUMNS
    ]
    .head(1)
    .copy()
)


# =========================================================
# 3. BUILD MLFLOW SERVING PAYLOAD
#
# MLflow supports dataframe_split format:
#
# {
#   "dataframe_split": {
#       "columns": [...],
#       "index": [...],
#       "data": [...]
#   }
# }
#
# Using DataFrame.to_json() also converts NumPy values
# into normal JSON-compatible values.
# =========================================================

dataframe_split = json.loads(
    sample.to_json(
        orient="split"
    )
)


payload = {
    "dataframe_split":
        dataframe_split
}


# =========================================================
# 4. CALL MLFLOW /invocations
# =========================================================

url = (
    f"{MLFLOW_SERVING_URL}"
    "/invocations"
)


print()
print(
    "Calling MLflow native serving..."
)

print(
    f"URL: {url}"
)


response = httpx.post(

    url,

    json=payload,

    timeout=30.0,
)


# Raise Python exception if HTTP status is not 2xx.
response.raise_for_status()


# =========================================================
# 5. DISPLAY RESULT
# =========================================================

print()

print(
    "MLflow Native Serving Response"
)

print(
    "========================================"
)

print(
    json.dumps(
        response.json(),
        indent=2,
    )
)