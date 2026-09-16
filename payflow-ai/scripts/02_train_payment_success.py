import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from dotenv import load_dotenv
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from payflow.data.load import load_transactions
from payflow.data.split import chronological_split
from payflow.data.validate import validate_transactions

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


# Normalize payment-status labels.
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


if df["target"].isna().any():
    raise ValueError(
        "Could not encode every transaction status."
    )


df["target"] = df["target"].astype(int)


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
# 7. CHECK CLASS DISTRIBUTIONS
# =========================================================

def validate_binary_classes(
    split_name: str,
    target: pd.Series,
) -> None:
    """
    Ensure each dataset split contains both classes.

    Required:
        FAILED  = 0
        SUCCESS = 1
    """

    classes = set(
        target.unique()
    )

    if classes != {0, 1}:
        raise ValueError(
            f"{split_name} must contain both classes. "
            f"Found: {sorted(classes)}"
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
# Important:
# Log the exact encoded target used during training.
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
# 9. DEFINE MODEL CANDIDATES
# =========================================================

models = {

    "dummy-baseline": (
        DummyClassifier(
            strategy="most_frequent",
        ),
        {
            "model_type":
                "DummyClassifier",

            "strategy":
                "most_frequent",
        },
    ),

    "logistic-regression-v1": (
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        {
            "model_type":
                "LogisticRegression",

            "max_iter":
                1000,

            "random_state":
                42,
        },
    ),
}


# =========================================================
# 10. TRAIN EACH MODEL IN ITS OWN MLFLOW RUN
# =========================================================

for (
    run_name,
    (
        classifier,
        model_params,
    ),
) in models.items():

    # -----------------------------------------------------
    # Keep preprocessing + model together.
    #
    # This is the object we will deploy later.
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
    # Start MLflow run.
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

                "environment":
                    "local",
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

                "feature_count":
                    len(FEATURE_COLUMNS),

                "target":
                    "target",

                "target_encoding":
                    "FAILED=0,SUCCESS=1",
            }
        )


        # =================================================
        # TRAIN
        # =================================================

        model_pipeline.fit(
            X_train,
            y_train,
        )


        # =================================================
        # VALIDATION PREDICTIONS
        # =================================================

        validation_predictions = (
            model_pipeline.predict(
                X_validation
            )
        )


        # -------------------------------------------------
        # Do NOT blindly assume predict_proba()[:, 1].
        #
        # Locate the probability column belonging to
        # class SUCCESS=1 explicitly.
        # -------------------------------------------------

        classifier_classes = (
            model_pipeline
            .named_steps["classifier"]
            .classes_
        )

        success_class_index = (
            list(
                classifier_classes
            ).index(1)
        )


        validation_probabilities = (
            model_pipeline.predict_proba(
                X_validation
            )[:, success_class_index]
        )


        # =================================================
        # METRICS
        # =================================================

        metrics = (
            calculate_payment_metrics(
                y_true=y_validation,
                y_pred=validation_predictions,
                success_probability=(
                    validation_probabilities
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

        confusion_matrix_fig = (
            create_confusion_matrix(
                y_validation,
                validation_predictions,
            )
        )


        mlflow.log_figure(
            confusion_matrix_fig,
            "plots/confusion_matrix.png",
        )


        plt.close(
            confusion_matrix_fig
        )


        # =================================================
        # ROC CURVE
        # =================================================

        roc_fig = (
            create_roc_curve(
                y_validation,
                validation_probabilities,
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
                validation_probabilities,
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
        # FEATURE / TARGET METADATA
        # =================================================

        mlflow.log_dict(
            {
                "features":
                    FEATURE_COLUMNS,

                "target":
                    "target",

                "source_target":
                    TARGET_COLUMN,

                "target_encoding":
                    {
                        "FAILED": 0,
                        "SUCCESS": 1,
                    },
            },
            "metadata/feature_definition.json",
        )


        # =================================================
        # MODEL SIGNATURE
        # =================================================

        signature_input = (
            X_train.head(100)
        )


        signature_predictions = (
            model_pipeline.predict(
                signature_input
            )
        )


        signature = infer_signature(
            signature_input,
            signature_predictions,
        )


        # Small valid sample stored with the MLflow model.
        input_example = (
            X_train.head(5)
        )


        # =================================================
        # LOG COMPLETE MODEL
        #
        # Includes:
        #
        # - preprocessing
        # - scaler
        # - one-hot encoder
        # - classifier
        # - signature
        # - input example
        # - dependency metadata
        # =================================================

        model_info = (
            mlflow_sklearn.log_model(
                sk_model=model_pipeline,
                name="payment_success_model",
                signature=signature,
                input_example=input_example,
            )
        )


        # =================================================
        # TERMINAL SUMMARY
        # =================================================

        print(
            "\n"
            "========================================"
        )

        print(
            f"Run Name : {run_name}"
        )

        print(
            f"Run ID   : {run.info.run_id}"
        )

        print(
            f"Model URI: {model_info.model_uri}"
        )


        for (
            metric_name,
            metric_value,
        ) in metrics.items():

            print(
                f"{metric_name}: "
                f"{metric_value:.6f}"
            )