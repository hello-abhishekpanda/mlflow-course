from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow

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


SELECTION_PATH = Path(
    "artifacts/selected_candidate.json"
)


THRESHOLD_POLICY_PATH = Path(
    "artifacts/threshold_policy.json"
)


FINAL_DECISION_PATH = Path(
    "artifacts/final_gate_decision.json"
)


# =========================================================
# 2. CONFIGURE MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)


client = MlflowClient()


# =========================================================
# 3. LOAD GOVERNANCE ARTIFACTS
# =========================================================

for required_path in [

    SELECTION_PATH,
    THRESHOLD_POLICY_PATH,
    FINAL_DECISION_PATH,

]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required governance artifact missing: "
            f"{required_path}"
        )


selection = json.loads(
    SELECTION_PATH.read_text(
        encoding="utf-8"
    )
)


threshold_policy = json.loads(
    THRESHOLD_POLICY_PATH.read_text(
        encoding="utf-8"
    )
)


final_decision = json.loads(
    FINAL_DECISION_PATH.read_text(
        encoding="utf-8"
    )
)


candidate = (
    selection["candidate"]
)


# =========================================================
# 4. HARD REGISTRY GATE
#
# This is extremely important.
#
# Registration MUST NOT happen just because somebody ran
# this Python script.
#
# The candidate must have passed the final quality gate.
# =========================================================

if not final_decision.get(
    "registry_eligible",
    False,
):

    raise RuntimeError(
        "\nMODEL REGISTRATION BLOCKED.\n\n"
        "final_gate_decision.json says:\n"
        "registry_eligible = false\n\n"
        "Fix the model-quality issue before registration."
    )


# =========================================================
# 5. VERIFY MODEL IDs MATCH
#
# Prevent stale governance files from accidentally
# registering the wrong model.
# =========================================================

expected_model_id = (
    candidate["model_id"]
)


if (
    final_decision[
        "candidate_model_id"
    ]
    != expected_model_id
):

    raise RuntimeError(
        "Final quality-gate decision belongs to another "
        "model."
    )


if (
    threshold_policy[
        "candidate_model_id"
    ]
    != expected_model_id
):

    raise RuntimeError(
        "Threshold policy belongs to another model."
    )


# =========================================================
# 6. SOURCE MODEL
#
# MLflow 3 logged-model URI:
#
#     models:/m-xxxxxxxx
#
# MLflow can register a Logged Model directly from this URI.
# =========================================================

source_model_uri = (
    candidate["model_uri"]
)


selected_threshold = float(
    threshold_policy[
        "selected_threshold"
    ]
)


# =========================================================
# 7. REGISTER MODEL VERSION
#
# If the registered model does not yet exist, MLflow
# creates it.
#
# If it already exists, MLflow creates the next version:
#
# Version 1
# Version 2
# Version 3
# ...
# =========================================================

print()

print(
    "Registering model..."
)


model_version = mlflow.register_model(

    model_uri=(
        source_model_uri
    ),

    name=(
        REGISTERED_MODEL_NAME
    ),

    tags={
        "project":
            "payflow-ai",

        "domain":
            "upi-payments",

        "quality_gate":
            "PASSED",

        "threshold_policy_version":
            threshold_policy[
                "policy_version"
            ],

        "failure_threshold":
            str(
                selected_threshold
            ),
    },
)


version = str(
    model_version.version
)


# =========================================================
# 8. REGISTERED MODEL-LEVEL TAGS
#
# These describe the complete model family, not one
# particular version.
# =========================================================

client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "project",
    "payflow-ai",
)


client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "domain",
    "upi-payment-success",
)


client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "currency",
    "INR",
)


# =========================================================
# 9. VERSION-SPECIFIC TAGS
#
# These belong to this exact registered version.
# =========================================================

version_tags = {

    "source_run_id":
        candidate[
            "run_id"
        ],

    "source_logged_model_id":
        candidate[
            "model_id"
        ],

    "quality_gate":
        "PASSED",

    "registry_eligible":
        "true",

    "threshold_policy_version":
        threshold_policy[
            "policy_version"
        ],

    "failure_threshold":
        str(
            selected_threshold
        ),

    "threshold_objective":
        threshold_policy[
            "objective"
        ],

    "threshold_optimized_on":
        threshold_policy[
            "optimized_on"
        ],
}


for (
    key,
    value,
) in version_tags.items():

    client.set_model_version_tag(

        name=(
            REGISTERED_MODEL_NAME
        ),

        version=(
            version
        ),

        key=(
            key
        ),

        value=(
            value
        ),
    )


# =========================================================
# 10. DETERMINE ALIAS
#
# First model ever registered:
#
#     Version 1
#         ↓
#     @champion
#
# Future approved candidate:
#
#     existing @champion
#         +
#     new Version N
#         ↓
#     @challenger
# =========================================================

registered_model = (
    client.get_registered_model(
        REGISTERED_MODEL_NAME
    )
)


existing_aliases = (
    registered_model.aliases
    or {}
)


if "champion" not in existing_aliases:

    assigned_alias = (
        "champion"
    )

else:

    assigned_alias = (
        "challenger"
    )


# =========================================================
# 11. ASSIGN MODEL ALIAS
# =========================================================

client.set_registered_model_alias(

    name=(
        REGISTERED_MODEL_NAME
    ),

    alias=(
        assigned_alias
    ),

    version=(
        version
    ),
)


# =========================================================
# 12. RECORD CURRENT LIFECYCLE ROLE
# =========================================================

client.set_model_version_tag(

    name=(
        REGISTERED_MODEL_NAME
    ),

    version=(
        version
    ),

    key=(
        "lifecycle_role"
    ),

    value=(
        assigned_alias
    ),
)


# =========================================================
# 13. TERMINAL SUMMARY
# =========================================================

print()
print(
    "========================================"
)

print(
    "MODEL REGISTRATION COMPLETE"
)

print(
    "========================================"
)

print(
    f"Registered Model : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Version          : "
    f"{version}"
)

print(
    f"Alias            : "
    f"@{assigned_alias}"
)

print(
    f"Source Model ID  : "
    f"{candidate['model_id']}"
)

print(
    f"Source Run ID    : "
    f"{candidate['run_id']}"
)

print(
    f"Failure Threshold: "
    f"{selected_threshold:.6f}"
)

print()

print(
    "Registered URI:"
)

print(
    f"models:/"
    f"{REGISTERED_MODEL_NAME}/"
    f"{version}"
)

print()

print(
    "Alias URI:"
)

print(
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{assigned_alias}"
)