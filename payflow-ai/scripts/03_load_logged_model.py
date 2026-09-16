import os

import mlflow
import mlflow.sklearn as mlflow_sklearn

from dotenv import load_dotenv


# =========================================================
# 1. LOAD ENVIRONMENT CONFIGURATION
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

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "payment_success_model",
)


# =========================================================
# 2. CONNECT TO MLFLOW TRACKING SERVER
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)


print(
    f"MLflow Tracking URI: {mlflow.get_tracking_uri()}"
)


# =========================================================
# 3. FIND THE EXPERIMENT
# =========================================================

experiment = (
    mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )
)


if experiment is None:
    raise RuntimeError(
        f"Experiment '{EXPERIMENT_NAME}' "
        "was not found."
    )


print(
    f"Experiment Name: {experiment.name}"
)

print(
    f"Experiment ID  : {experiment.experiment_id}"
)


# =========================================================
# 4. FIND THE LATEST LOGGED MODEL
#
# MLflow 3 stores logged models as first-class resources.
#
# We search for:
#
#     payment_success_model
#
# inside:
#
#     payflow-payment-success
#
# and select the most recently created model.
# =========================================================

logged_models = (
    mlflow.search_logged_models(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            f"name = '{MODEL_NAME}'"
        ),
        order_by=[
            {
                "field_name":
                    "creation_time",

                "ascending":
                    False,
            }
        ],
        max_results=1,
        output_format="list",
    )
)


if not logged_models:
    raise RuntimeError(
        f"No logged model named "
        f"'{MODEL_NAME}' was found "
        f"in experiment "
        f"'{EXPERIMENT_NAME}'."
    )


# Latest model.
logged_model = logged_models[0]


# =========================================================
# 5. BUILD THE MLFLOW 3 MODEL URI
#
# MLflow 3 preferred format:
#
#     models:/m-xxxxxxxxxxxxxxxx
# =========================================================

MODEL_URI = (
    f"models:/{logged_model.model_id}"
)


print()
print(
    "Selected Logged Model"
)
print(
    "----------------------------------------"
)

print(
    f"Model Name : {logged_model.name}"
)

print(
    f"Model ID   : {logged_model.model_id}"
)

print(
    f"Source Run : {logged_model.source_run_id}"
)

print(
    f"Model URI  : {MODEL_URI}"
)


# =========================================================
# 6. LOAD THE EXACT SCIKIT-LEARN PIPELINE
#
# This loads:
#
# - preprocessing
# - StandardScaler
# - OneHotEncoder
# - classifier
#
# because we logged the complete sklearn Pipeline.
# =========================================================

model = mlflow_sklearn.load_model(
    MODEL_URI
)


# =========================================================
# 7. VERIFY
# =========================================================

print()
print(
    "Model loaded successfully."
)

print(
    f"Python Type: {type(model)}"
)


# Optional:
# Display sklearn Pipeline steps.
if hasattr(model, "named_steps"):

    print()
    print(
        "Pipeline Steps:"
    )

    assert model is not None
    for step_name in (
        model.named_steps
    ):

        print(
            f"  - {step_name}"
        )