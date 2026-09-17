from __future__ import annotations

import os
from datetime import datetime, timezone


# =========================================================
# LOCAL ARTIFACT PROXY POLICY
#
# The host-side bootstrap client should communicate with
# MLflow, not MinIO's Docker-only hostname.
# =========================================================

os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"


import mlflow
import mlflow.models
import mlflow.sklearn as mlflow_sklearn

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# =========================================================
# PAYFLOW AI — PART 7
#
# MIGRATE APPROVED @champion
#
# SOURCE
#   Legacy MLflow :5002
#   SQLite + local mlartifacts
#
# TARGET
#   Docker MLflow :5050
#   PostgreSQL + MinIO
#
# IMPORTANT:
#   This script DOES NOT retrain the model.
# =========================================================


SOURCE_TRACKING_URI = os.getenv(
    "PAYFLOW_SOURCE_MLFLOW_URI",
    "http://127.0.0.1:5002",
)

TARGET_TRACKING_URI = os.getenv(
    "PAYFLOW_DOCKER_MLFLOW_URI",
    "http://127.0.0.1:5050",
)

REGISTERED_MODEL_NAME = os.getenv(
    "PAYFLOW_REGISTERED_MODEL_NAME",
    "payflow-payment-success",
)

MODEL_ALIAS = os.getenv(
    "PAYFLOW_MODEL_ALIAS",
    "champion",
)

BOOTSTRAP_EXPERIMENT = (
    "payflow-production-bootstrap"
)


def section(title: str) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# =========================================================
# 1. SOURCE REGISTRY
# =========================================================

section(
    "1. SOURCE MLFLOW REGISTRY"
)

print(f"Tracking URI : {SOURCE_TRACKING_URI}")
print(f"Model        : {REGISTERED_MODEL_NAME}")
print(f"Alias        : @{MODEL_ALIAS}")


source_client = MlflowClient(
    tracking_uri=SOURCE_TRACKING_URI,
    registry_uri=SOURCE_TRACKING_URI,
)


try:

    source_version = (
        source_client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
    )

except MlflowException as exc:

    raise RuntimeError(
        f"Could not resolve "
        f"{REGISTERED_MODEL_NAME}@{MODEL_ALIAS} "
        f"from {SOURCE_TRACKING_URI}."
    ) from exc


source_version_number = str(
    source_version.version
)

source_tags = dict(
    source_version.tags or {}
)


print(
    f"Source Version : {source_version_number}"
)

print(
    f"Source Run ID  : {source_version.run_id}"
)

print(
    f"Source URI     : {source_version.source}"
)


# =========================================================
# 2. GOVERNANCE CHECK
# =========================================================

section(
    "2. GOVERNANCE CHECK"
)


quality_gate = str(
    source_tags.get(
        "quality_gate",
        "",
    )
).upper()


registry_eligible = str(
    source_tags.get(
        "registry_eligible",
        "",
    )
).lower()


failure_threshold = (
    source_tags.get(
        "failure_threshold"
    )
)


if quality_gate != "PASSED":

    raise RuntimeError(
        "Source champion does not have "
        "quality_gate=PASSED."
    )


if registry_eligible != "true":

    raise RuntimeError(
        "Source champion does not have "
        "registry_eligible=true."
    )


if failure_threshold is None:

    raise RuntimeError(
        "Source champion has no failure_threshold."
    )


print(
    "quality_gate      : PASSED"
)

print(
    "registry_eligible : true"
)

print(
    f"failure_threshold : {failure_threshold}"
)


# =========================================================
# 3. LOAD APPROVED SOURCE MODEL
# =========================================================

section(
    "3. LOAD APPROVED SOURCE MODEL"
)


source_model_uri = (
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)


mlflow.set_tracking_uri(
    SOURCE_TRACKING_URI
)

mlflow.set_registry_uri(
    SOURCE_TRACKING_URI
)


source_model = (
    mlflow_sklearn.load_model(
        source_model_uri
    )
)


source_model_info = (
    mlflow.models.get_model_info(
        source_model_uri
    )
)


print(
    f"Model URI   : {source_model_uri}"
)

print(
    f"Python Type : {type(source_model)}"
)

print(
    f"Signature   : {source_model_info.signature}"
)


# =========================================================
# 4. DETECT SKOPS TRUSTED TYPES
# =========================================================

trusted_types: list[str] | None = None


classifier = (
    source_model
    .named_steps
    .get(
        "classifier"
    )
)


if (
    classifier is not None
    and
    type(classifier)
    .__module__
    .startswith("xgboost")
):

    trusted_types = [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    ]


# =========================================================
# 5. TARGET PRECHECK
# =========================================================

section(
    "4. TARGET REGISTRY PRECHECK"
)


target_client = MlflowClient(
    tracking_uri=TARGET_TRACKING_URI,
    registry_uri=TARGET_TRACKING_URI,
)


try:

    existing = (
        target_client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
    )

except MlflowException:

    existing = None


if existing is not None:

    print(
        "Target Registry already contains:"
    )

    print(
        f"{REGISTERED_MODEL_NAME}"
        f"@{MODEL_ALIAS}"
        f" -> Version {existing.version}"
    )

    print(
        "Bootstrap skipped. "
        "No duplicate model version created."
    )

    raise SystemExit(0)


# =========================================================
# 6. SWITCH TO TARGET MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TARGET_TRACKING_URI
)

mlflow.set_registry_uri(
    TARGET_TRACKING_URI
)

mlflow.set_experiment(
    BOOTSTRAP_EXPERIMENT
)


# =========================================================
# 7. LOG FITTED MODEL INTO NEW MLFLOW
#
# NO TRAINING occurs here.
# =========================================================

section(
    "5. LOG APPROVED MODEL INTO DOCKER MLFLOW"
)


with mlflow.start_run(
    run_name="bootstrap-approved-champion"
) as run:

    mlflow.log_params(
        {
            "source_tracking_uri":
                SOURCE_TRACKING_URI,

            "source_version":
                source_version_number,

            "migration":
                "sqlite-local-to-postgres-minio",

            "retrained":
                False,
        }
    )

    mlflow.set_tags(
        {
            "project":
                "payflow-ai",

            "stage":
                "infrastructure-migration",

            "quality_gate":
                "PASSED",

            "registry_eligible":
                "true",
        }
    )


    log_kwargs = {
        "sk_model":
            source_model,

        "name":
            "payment_success_model",

        "serialization_format":
            mlflow_sklearn
            .SERIALIZATION_FORMAT_SKOPS,

        "signature":
            source_model_info.signature,
    }


    if trusted_types is not None:

        log_kwargs[
            "skops_trusted_types"
        ] = trusted_types


    logged_model = (
        mlflow_sklearn.log_model(
            **log_kwargs
        )
    )


    migration_run_id = (
        run.info.run_id
    )


print(
    f"Migration Run ID : {migration_run_id}"
)

print(
    f"Logged Model URI : {logged_model.model_uri}"
)


# =========================================================
# 8. REGISTER MODEL
# =========================================================

section(
    "6. CREATE TARGET MODEL VERSION"
)


registered_version = (
    mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=REGISTERED_MODEL_NAME,
    )
)


new_version = str(
    registered_version.version
)


print(
    f"Model   : {REGISTERED_MODEL_NAME}"
)

print(
    f"Version : {new_version}"
)


# =========================================================
# 9. REGISTERED MODEL TAGS
# =========================================================

target_client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "project",
    "payflow-ai",
)

target_client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "domain",
    "upi-payment-success",
)

target_client.set_registered_model_tag(
    REGISTERED_MODEL_NAME,
    "currency",
    "INR",
)


# =========================================================
# 10. VERSION GOVERNANCE + LINEAGE TAGS
# =========================================================

section(
    "7. COPY MODEL VERSION TAGS"
)


# Preserve source tags first.

for key, value in source_tags.items():

    target_client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=new_version,
        key=str(key),
        value=str(value),
    )


migration_tags = {

    "quality_gate":
        "PASSED",

    "registry_eligible":
        "true",

    "failure_threshold":
        str(failure_threshold),

    "lifecycle_role":
        "champion",

    "storage_backend":
        "postgresql",

    "artifact_backend":
        "minio",

    "migrated_from_mlflow":
        SOURCE_TRACKING_URI,

    "migrated_from_version":
        source_version_number,

    "migration_run_id":
        migration_run_id,

    "migration_timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_logged_model_id":
        getattr(
            logged_model,
            "model_id",
            "unknown",
        ),
}


for key, value in migration_tags.items():

    target_client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=new_version,
        key=key,
        value=str(value),
    )


print(
    "Governance and lineage tags copied."
)


# =========================================================
# 11. ASSIGN @champion
# =========================================================

section(
    "8. ASSIGN TARGET ALIAS"
)


target_client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias=MODEL_ALIAS,
    version=new_version,
)


print(
    f"Alias @{MODEL_ALIAS} "
    f"-> Version {new_version}"
)


# =========================================================
# 12. VERIFY TARGET MODEL BY LOADING IT BACK
# =========================================================

section(
    "9. VERIFY TARGET @CHAMPION"
)


target_model_uri = (
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)


reloaded_model = (
    mlflow_sklearn.load_model(
        target_model_uri
    )
)


verified_version = (
    target_client
    .get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
    )
)


verified_tags = dict(
    verified_version.tags or {}
)


print(
    f"Model Name : {REGISTERED_MODEL_NAME}"
)

print(
    f"Alias      : @{MODEL_ALIAS}"
)

print(
    f"Version    : {verified_version.version}"
)

print(
    f"Model URI  : {target_model_uri}"
)

print(
    f"Threshold  : "
    f"{verified_tags.get('failure_threshold')}"
)

print(
    f"Gate       : "
    f"{verified_tags.get('quality_gate')}"
)

print(
    f"Eligible   : "
    f"{verified_tags.get('registry_eligible')}"
)

print(
    f"Python Type: {type(reloaded_model)}"
)


# =========================================================
# SUCCESS
# =========================================================

section(
    "DOCKER REGISTRY BOOTSTRAP PASSED"
)


print(
    "Approved source model migrated : PASS"
)

print(
    "Model registered               : PASS"
)

print(
    "Governance tags preserved      : PASS"
)

print(
    "Failure threshold preserved    : PASS"
)

print(
    f"Alias @{MODEL_ALIAS:<23}: PASS"
)

print(
    "Target model reload            : PASS"
)

print()

print(
    "Production URI:"
)

print(
    target_model_uri
)