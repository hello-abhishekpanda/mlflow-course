from __future__ import annotations

import os

import mlflow
import mlflow.sklearn as mlflow_sklearn

from dotenv import load_dotenv

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.pipeline import Pipeline

from payflow.data.load import (
    load_transactions,
)

from payflow.data.split import (
    chronological_split,
)

from payflow.data.validate import (
    validate_transactions,
)

from payflow.features.payment_success import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from payflow.features.pipeline import (
    build_preprocessor,
)

from payflow.models.metrics import (
    calculate_payment_metrics,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

DATA_PATH = os.getenv(
    "PAYFLOW_DATA_PATH",
    "data/raw/upi_transactions_2024.csv",
)


mlflow.set_tracking_uri(
    TRACKING_URI
)


mlflow.set_experiment(
    "payflow-autolog-demo"
)


# =========================================================
# 2. ENABLE SKLEARN AUTOLOGGING
#
# pos_label=0:
#     FAILED is our positive business event for
#     classification metric reporting.
#
# log_input_examples=True:
#     Store a representative model input.
#
# log_model_signatures=True:
#     Store model contract.
#
# log_models=True:
#     Store fitted Pipeline.
# =========================================================

mlflow_sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    log_post_training_metrics=True,
    pos_label=0,
)


# =========================================================
# 3. LOAD DATA
# =========================================================

df = load_transactions(
    DATA_PATH
)

validate_transactions(
    df
)


df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.upper()
)


df["target"] = (
    df[TARGET_COLUMN]
    .map(
        {
            "FAILED": 0,
            "SUCCESS": 1,
        }
    )
    .astype(int)
)


# =========================================================
# 4. SPLIT DATA
# =========================================================

(
    train_df,
    validation_df,
    _,
) = chronological_split(
    df
)


X_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .copy()
)

y_train = (
    train_df["target"]
    .copy()
)


X_validation = (
    validation_df[
        FEATURE_COLUMNS
    ]
    .copy()
)

y_validation = (
    validation_df["target"]
    .copy()
)


# =========================================================
# 5. BUILD MODEL PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# =========================================================
# 6. START ONE MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name="random-forest-autolog-demo"
):

    # -----------------------------------------------------
    # Keep business metadata manual.
    # -----------------------------------------------------

    mlflow.set_tags(
        {
            "project":
                "payflow-ai",

            "purpose":
                "autolog-demo",

            "currency":
                "INR",

            "problem":
                "payment-success",
        }
    )


    # -----------------------------------------------------
    # Calling fit() triggers MLflow sklearn autologging.
    # -----------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )


    # -----------------------------------------------------
    # Our custom validation metrics still matter.
    # -----------------------------------------------------

    predictions = (
        pipeline.predict(
            X_validation
        )
    )


    classes = (
        pipeline
        .named_steps["classifier"]
        .classes_
    )


    success_index = (
        list(classes)
        .index(1)
    )


    probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, success_index]
    )


    metrics = (
        calculate_payment_metrics(
            y_true=y_validation,
            y_pred=predictions,
            success_probability=probabilities,
        )
    )


    # -----------------------------------------------------
    # These are PayFlow-specific validation metrics,
    # so we explicitly log them.
    # -----------------------------------------------------

    mlflow.log_metrics(
        {
            f"val_{name}": value
            for (
                name,
                value,
            )
            in metrics.items()
        }
    )


print(
    "Autologging experiment completed."
)