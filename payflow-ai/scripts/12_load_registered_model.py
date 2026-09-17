from __future__ import annotations

import os

import mlflow
import mlflow.sklearn as mlflow_sklearn

from dotenv import load_dotenv
from mlflow import MlflowClient


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


REGISTERED_MODEL_NAME = os.getenv(
    "PAYFLOW_REGISTERED_MODEL_NAME",
    "payflow-payment-success",
)


MODEL_ALIAS = os.getenv(
    "PAYFLOW_MODEL_ALIAS",
    "champion",
)


# =========================================================
# 2. CONFIGURE MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)


client = MlflowClient()


# =========================================================
# 3. LOOK UP ALIAS METADATA
#
# Example:
#
#     champion
#
# may currently point to:
#
#     Version 3
#
# Later it could point to Version 4 without this inference
# code changing at all.
# =========================================================

model_version = (
    client.get_model_version_by_alias(

        name=(
            REGISTERED_MODEL_NAME
        ),

        alias=(
            MODEL_ALIAS
        ),
    )
)


# =========================================================
# 4. READ FAILURE THRESHOLD FROM VERSION TAGS
#
# The threshold belongs to the version.
#
# Never hard-code:
#
#     0.0834
#
# inside the API.
# =========================================================

threshold_value = (
    model_version.tags.get(
        "failure_threshold"
    )
)


if threshold_value is None:

    raise RuntimeError(
        "Registered model version has no "
        "'failure_threshold' tag."
    )


failure_threshold = float(
    threshold_value
)


# =========================================================
# 5. BUILD ALIAS URI
# =========================================================

MODEL_URI = (
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)


# =========================================================
# 6. LOAD MODEL
# =========================================================

model = (
    mlflow_sklearn.load_model(
        MODEL_URI
    )
)


# =========================================================
# 7. VERIFY
# =========================================================

print()
print(
    "========================================"
)

print(
    "REGISTERED MODEL LOADED"
)

print(
    "========================================"
)

print(
    f"Model Name : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Alias      : "
    f"@{MODEL_ALIAS}"
)

print(
    f"Version    : "
    f"{model_version.version}"
)

print(
    f"Model URI  : "
    f"{MODEL_URI}"
)

print(
    f"Threshold  : "
    f"{failure_threshold:.6f}"
)

print(
    f"Python Type: "
    f"{type(model)}"
)