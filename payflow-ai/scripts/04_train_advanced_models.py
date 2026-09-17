from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from dotenv import load_dotenv
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from payflow.data.load import load_transactions
from payflow.data.split import chronological_split
from payflow.data.validate import validate_transactions

from payflow.features.payment_success import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from payflow.features.pipeline import build_preprocessor

from payflow.models.feature_importance import (
    create_feature_importance_plot,
    extract_feature_importance,
)

from payflow.models.metrics import (
    calculate_payment_metrics,
)

from payflow.models.plots import (
    create_confusion_matrix,
    create_failure_pr_curve,
    create_roc_curve,
)


# =========================================================
# 1. LOAD CONFIGURATION
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

DATA_PATH = Path(
    os.getenv(
        "PAYFLOW_DATA_PATH",
        "data/raw/upi_transactions_2024.csv",
    )
)


# =========================================================
# 2. CONFIGURE MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# =========================================================
# 3. LOAD + VALIDATE DATA
# =========================================================

df = load_transactions(
    DATA_PATH
)

validate_transactions(
    df
)


# Normalize the payment-status labels.
df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.upper()
)


# =========================================================
# 4. ENCODE TARGET
#
# FAILED  = 0
# SUCCESS = 1
# =========================================================

df["target"] = (
    df[TARGET_COLUMN]
    .map(
        {
            "FAILED": 0,
            "SUCCESS": 1,
        }
    )
)


# If a value could not be mapped, stop immediately.
if df["target"].isna().any():

    bad_values = sorted(
        df.loc[
            df["target"].isna(),
            TARGET_COLUMN,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    raise ValueError(
        "Could not encode every transaction status. "
        f"Unexpected values: {bad_values}"
    )


df["target"] = (
    df["target"]
    .astype(int)
)


# =========================================================
# 5. CHRONOLOGICAL SPLIT
# =========================================================

(
    train_df,
    validation_df,
    test_df,
) = chronological_split(
    df
)


# =========================================================
# 6. FEATURES + LABELS
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
].copy()

y_train = train_df[
    "target"
].copy()


X_validation = validation_df[
    FEATURE_COLUMNS
].copy()

y_validation = validation_df[
    "target"
].copy()


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    "target"
].copy()


# =========================================================
# 7. VERIFY BOTH CLASSES EXIST
# =========================================================

def validate_binary_classes(
    split_name: str,
    target: pd.Series,
) -> None:
    """
    Ensure a split contains both payment outcomes.

    0 = FAILED
    1 = SUCCESS

    Metrics such as ROC-AUC require both classes.
    """

    classes = set(
        target.unique()
    )

    if classes != {0, 1}:
        raise ValueError(
            f"{split_name} must contain both classes "
            f"{{0, 1}}. Found: {sorted(classes)}"
        )


validate_binary_classes(
    "Training split",
    y_train,
)

validate_binary_classes(
    "Validation split",
    y_validation,
)

validate_binary_classes(
    "Test split",
    y_test,
)


# =========================================================
# 8. CREATE MLFLOW DATASET
#
# Log the exact features + encoded target used for training.
# =========================================================

training_frame = (
    X_train.copy()
)

training_frame["target"] = (
    y_train.to_numpy()
)


training_dataset = from_pandas(
    training_frame,
    source=str(DATA_PATH),
    targets="target",
    name="upi-transactions-2024-training",
)


# =========================================================
# 9. SKOPS TRUST CONFIGURATION FOR XGBOOST
#
# MLflow 3.16 uses SKOPS by default when
# mlflow.sklearn.log_model() is called.
#
# SKOPS does not automatically trust third-party Python
# classes. Our sklearn Pipeline contains XGBClassifier,
# therefore we explicitly trust these two XGBoost types.
#
# This fixes:
#
# UntrustedTypesFoundException:
#   xgboost.core.Booster
#   xgboost.sklearn.XGBClassifier
#
# MLflow stores this trusted-type list in the model's
# sklearn flavor configuration, so normal load_model()
# works later as well.
# =========================================================

XGBOOST_SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


# =========================================================
# 10. DEFINE ADVANCED MODEL CANDIDATES
# =========================================================

models: dict[
    str,
    dict[str, Any],
] = {

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    "random-forest-v1": {

        "classifier": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),

        "params": {
            "model_type":
                "RandomForestClassifier",

            "n_estimators":
                300,

            "max_depth":
                "None",

            "min_samples_leaf":
                2,

            "random_state":
                42,
        },

        # Random Forest works directly with SKOPS.
        "skops_trusted_types":
            None,
    },


    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

    "xgboost-v1": {

        "classifier": (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        ),

        "params": {
            "model_type":
                "XGBClassifier",

            "n_estimators":
                300,

            "max_depth":
                6,

            "learning_rate":
                0.05,

            "subsample":
                0.80,

            "colsample_bytree":
                0.80,

            "objective":
                "binary:logistic",

            "eval_metric":
                "logloss",

            "tree_method":
                "hist",

            "random_state":
                42,
        },

        # -------------------------------------------------
        # FIX FOR THE ERROR YOU RECEIVED.
        # -------------------------------------------------

        "skops_trusted_types":
            XGBOOST_SKOPS_TRUSTED_TYPES,
    },
}


# =========================================================
# 11. OPTIONAL MODEL FILTER
#
# Default:
#     run Random Forest + XGBoost
#
# If Random Forest already succeeded and only XGBoost
# failed, run:
#
# PAYFLOW_MODELS=xgboost-v1 \
# python scripts/04_train_advanced_models.py
#
# This prevents another duplicate Random Forest run.
# =========================================================

requested_models_raw = os.getenv(
    "PAYFLOW_MODELS",
    ",".join(
        models.keys()
    ),
)


requested_models = {
    name.strip()

    for name
    in requested_models_raw.split(",")

    if name.strip()
}


unknown_models = (
    requested_models
    - set(
        models.keys()
    )
)


if unknown_models:
    raise ValueError(
        "Unknown model name(s) in PAYFLOW_MODELS: "
        f"{sorted(unknown_models)}. "
        "Available models: "
        f"{sorted(models.keys())}"
    )


models_to_run = {
    name: config

    for (
        name,
        config,
    )
    in models.items()

    if name in requested_models
}


if not models_to_run:
    raise ValueError(
        "No models selected. "
        "Set PAYFLOW_MODELS to a valid model name."
    )


print()

print(
    "Models selected for this execution:"
)


for selected_model_name in models_to_run:

    print(
        f"  - {selected_model_name}"
    )


# =========================================================
# 12. TRAIN EACH MODEL
# =========================================================

for (
    run_name,
    model_config,
) in models_to_run.items():

    classifier = (
        model_config[
            "classifier"
        ]
    )

    model_params = (
        model_config[
            "params"
        ]
    )

    skops_trusted_types = (
        model_config[
            "skops_trusted_types"
        ]
    )


    print()

    print(
        "========================================"
    )

    print(
        f"Training: {run_name}"
    )

    print(
        "========================================"
    )


    # -----------------------------------------------------
    # Build a fresh pipeline for each model.
    #
    # The final MLflow model will contain:
    #
    # preprocessing
    #     +
    # classifier
    #
    # This is essential because inference must use exactly
    # the same transformations as training.
    # -----------------------------------------------------

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


    # -----------------------------------------------------
    # ONE CANDIDATE = ONE MLFLOW RUN
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ) as run:


        # =================================================
        # DATASET LINEAGE
        # =================================================

        mlflow.log_input(
            training_dataset,
            context="training",
        )


        # =================================================
        # TAGS
        # =================================================

        mlflow.set_tags(
            {
                "project":
                    "payflow-ai",

                "domain":
                    "upi-payments",

                "currency":
                    "INR",

                "problem":
                    "payment-success",

                "feature_set":
                    "v1",

                "split_strategy":
                    "chronological",

                "experiment_stage":
                    "candidate-comparison",

                "environment":
                    "local",

                "model_serialization":
                    "skops",
            }
        )


        # =================================================
        # PARAMETERS
        # =================================================

        mlflow.log_params(
            model_params
        )


        mlflow.log_params(
            {
                "train_rows":
                    len(X_train),

                "validation_rows":
                    len(X_validation),

                "test_rows":
                    len(X_test),

                "raw_feature_count":
                    len(FEATURE_COLUMNS),

                "target":
                    "target",

                "target_encoding":
                    "FAILED=0,SUCCESS=1",
            }
        )


        # =================================================
        # TRAINING
        # =================================================

        training_start = (
            time.perf_counter()
        )


        model_pipeline.fit(
            X_train,
            y_train,
        )


        training_seconds = (
            time.perf_counter()
            - training_start
        )


        mlflow.log_metric(
            "train_seconds",
            float(
                training_seconds
            ),
        )


        # =================================================
        # VALIDATION PREDICTIONS
        # =================================================

        prediction_start = (
            time.perf_counter()
        )


        validation_predictions = (
            model_pipeline.predict(
                X_validation
            )
        )


        probability_matrix = (
            model_pipeline.predict_proba(
                X_validation
            )
        )


        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )


        # -------------------------------------------------
        # Do not blindly assume [:, 1].
        #
        # Find the actual probability column for SUCCESS=1.
        # -------------------------------------------------

        classifier_classes = (
            model_pipeline
            .named_steps[
                "classifier"
            ]
            .classes_
        )


        success_class_index = (
            list(
                classifier_classes
            )
            .index(1)
        )


        success_probabilities = (
            probability_matrix[
                :,
                success_class_index,
            ]
        )


        # =================================================
        # INFERENCE LATENCY
        # =================================================

        validation_rows = (
            len(
                X_validation
            )
        )


        predict_ms_per_row = (
            prediction_seconds
            / validation_rows
            * 1000.0
        )


        mlflow.log_metric(
            "validation_predict_seconds",
            float(
                prediction_seconds
            ),
        )


        mlflow.log_metric(
            "predict_ms_per_row",
            float(
                predict_ms_per_row
            ),
        )


        # =================================================
        # VALIDATION METRICS
        # =================================================

        metrics = (
            calculate_payment_metrics(
                y_true=y_validation,
                y_pred=(
                    validation_predictions
                ),
                success_probability=(
                    success_probabilities
                ),
            )
        )


        mlflow.log_metrics(
            {
                f"val_{metric_name}":
                    metric_value

                for (
                    metric_name,
                    metric_value,
                )
                in metrics.items()
            }
        )


        # =================================================
        # CONFUSION MATRIX
        # =================================================

        confusion_fig = (
            create_confusion_matrix(
                y_validation,
                validation_predictions,
            )
        )


        mlflow.log_figure(
            confusion_fig,
            "plots/confusion_matrix.png",
        )


        plt.close(
            confusion_fig
        )


        # =================================================
        # ROC CURVE
        # =================================================

        roc_fig = (
            create_roc_curve(
                y_validation,
                success_probabilities,
            )
        )


        mlflow.log_figure(
            roc_fig,
            "plots/roc_curve.png",
        )


        plt.close(
            roc_fig
        )


        # =================================================
        # FAILURE PRECISION-RECALL CURVE
        # =================================================

        pr_fig = (
            create_failure_pr_curve(
                y_validation,
                success_probabilities,
            )
        )


        mlflow.log_figure(
            pr_fig,
            "plots/failure_pr_curve.png",
        )


        plt.close(
            pr_fig
        )


        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        importance_df = (
            extract_feature_importance(
                model_pipeline
            )
        )


        if importance_df is not None:

            # ---------------------------------------------
            # Save the complete importance table.
            # ---------------------------------------------

            importance_path = (
                Path(
                    "artifacts"
                )
                /
                (
                    f"{run_name}_"
                    "feature_importance.csv"
                )
            )


            importance_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            importance_df.to_csv(
                importance_path,
                index=False,
            )


            mlflow.log_artifact(
                str(
                    importance_path
                ),
                artifact_path=(
                    "feature_importance"
                ),
            )


            # ---------------------------------------------
            # Top-20 feature-importance plot.
            # ---------------------------------------------

            importance_fig = (
                create_feature_importance_plot(
                    importance_df,
                    top_n=20,
                )
            )


            mlflow.log_figure(
                importance_fig,
                (
                    "plots/"
                    "feature_importance_top20.png"
                ),
            )


            plt.close(
                importance_fig
            )


        # =================================================
        # FEATURE METADATA
        # =================================================

        mlflow.log_dict(
            {
                "raw_features":
                    FEATURE_COLUMNS,

                "raw_feature_count":
                    len(
                        FEATURE_COLUMNS
                    ),

                "target":
                    "target",

                "source_target":
                    TARGET_COLUMN,

                "target_encoding":
                    {
                        "FAILED":
                            0,

                        "SUCCESS":
                            1,
                    },
            },

            (
                "metadata/"
                "feature_definition.json"
            ),
        )


        # =================================================
        # MODEL SIGNATURE
        # =================================================

        signature_input = (
            X_train.head(
                100
            )
        )


        signature_predictions = (
            model_pipeline.predict(
                signature_input
            )
        )


        signature = (
            infer_signature(
                signature_input,
                signature_predictions,
            )
        )


        # Small valid inference sample saved with model.
        input_example = (
            X_train.head(
                5
            )
        )


        # =================================================
        # LOG COMPLETE PIPELINE
        #
        # THIS IS THE IMPORTANT FIX.
        #
        # MLflow 3.16 defaults mlflow.sklearn logging to
        # the SKOPS serialization format.
        #
        # Random Forest:
        #
        #     trusted types = None
        #
        # XGBoost:
        #
        #     trusted types =
        #
        #     xgboost.core.Booster
        #     xgboost.sklearn.XGBClassifier
        #
        # We keep the safer SKOPS format instead of
        # switching the entire project to cloudpickle.
        # =================================================

        model_info = (
            mlflow_sklearn.log_model(

                sk_model=(
                    model_pipeline
                ),

                name=(
                    "payment_success_model"
                ),

                serialization_format=(
                    mlflow_sklearn
                    .SERIALIZATION_FORMAT_SKOPS
                ),

                skops_trusted_types=(
                    skops_trusted_types
                ),

                signature=(
                    signature
                ),

                input_example=(
                    input_example
                ),
            )
        )


        # =================================================
        # TERMINAL SUMMARY
        # =================================================

        print()

        print(
            "MLflow model logged successfully."
        )


        print(
            f"Run Name : {run_name}"
        )


        print(
            f"Run ID   : {run.info.run_id}"
        )


        print(
            f"Model ID : {model_info.model_id}"
        )


        print(
            f"Model URI: {model_info.model_uri}"
        )


        print(
            "Serialization: "
            f"{mlflow_sklearn.SERIALIZATION_FORMAT_SKOPS}"
        )


        print(
            f"Train sec: "
            f"{training_seconds:.4f}"
        )


        print(
            "Predict ms/row: "
            f"{predict_ms_per_row:.6f}"
        )


        for (
            metric_name,
            metric_value,
        ) in metrics.items():

            print(
                f"{metric_name}: "
                f"{metric_value:.6f}"
            )


        print(
            "========================================"
        )