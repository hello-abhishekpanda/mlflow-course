from __future__ import annotations

import os
import sys

import mlflow

from dotenv import load_dotenv
from mlflow import MlflowClient


# =========================================================
# 1. LOAD CONFIGURATION
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


# =========================================================
# 2. CONFIGURE MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)


client = MlflowClient()


print()
print(
    "========================================"
)

print(
    "PAYFLOW MODEL PROMOTION"
)

print(
    "========================================"
)

print(
    f"Tracking URI     : {TRACKING_URI}"
)

print(
    f"Registered Model : {REGISTERED_MODEL_NAME}"
)


# =========================================================
# 3. LOAD REGISTERED MODEL
# =========================================================

try:

    registered_model = (
        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )
    )

except Exception as error:

    print()
    print(
        "Registered model could not be loaded."
    )

    print(
        error
    )

    sys.exit(1)


# =========================================================
# 4. READ CURRENT ALIASES
#
# Example:
#
# {
#     "champion": "1",
#     "challenger": "2"
# }
# =========================================================

aliases = (
    registered_model.aliases
    or {}
)


current_champion_version = (
    aliases.get(
        "champion"
    )
)


challenger_version = (
    aliases.get(
        "challenger"
    )
)


# =========================================================
# 5. DISPLAY CURRENT STATE
# =========================================================

print()

print(
    "Current Registry State"
)

print(
    "----------------------------------------"
)


if current_champion_version is None:

    print(
        "@champion   : NOT ASSIGNED"
    )

else:

    print(
        f"@champion   : Version "
        f"{current_champion_version}"
    )


if challenger_version is None:

    print(
        "@challenger : NOT ASSIGNED"
    )

else:

    print(
        f"@challenger : Version "
        f"{challenger_version}"
    )


# =========================================================
# 6. NO CHALLENGER IS NOT AN ERROR
#
# This is a perfectly valid registry state.
#
# Example:
#
# Version 1
#     ↓
# @champion
#
# No newer approved model exists yet.
# =========================================================

if challenger_version is None:

    print()
    print(
        "========================================"
    )

    print(
        "NO CHALLENGER AVAILABLE"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Nothing will be promoted."
    )

    print()

    print(
        "Current production model remains:"
    )

    if current_champion_version is not None:

        print(
            f"Version {current_champion_version} "
            "(@champion)"
        )

    else:

        print(
            "No champion is currently assigned."
        )

    print()

    print(
        "A challenger will appear after:"
    )

    print()
    print(
        "1. A new model is trained"
    )

    print(
        "2. It passes validation"
    )

    print(
        "3. It passes threshold optimization"
    )

    print(
        "4. It passes the final quality gate"
    )

    print(
        "5. scripts/11_register_candidate.py "
        "registers the new version"
    )

    print()

    print(
        "Promotion skipped safely."
    )

    sys.exit(0)


# =========================================================
# 7. SAFETY CHECK
#
# Champion and challenger must never be the same version.
# =========================================================

if (
    current_champion_version
    == challenger_version
):

    print()
    print(
        "Promotion blocked."
    )

    print(
        "Champion and challenger point to the "
        "same model version."
    )

    sys.exit(1)


# =========================================================
# 8. LOAD CHALLENGER DETAILS
#
# This gives us metadata before promotion.
# =========================================================

challenger_model_version = (
    client.get_model_version(

        name=(
            REGISTERED_MODEL_NAME
        ),

        version=(
            challenger_version
        ),
    )
)


# =========================================================
# 9. VERIFY CHALLENGER PASSED QUALITY GATE
#
# We do not trust the alias by itself.
#
# The registered version should contain:
#
# quality_gate = PASSED
# registry_eligible = true
# =========================================================

quality_gate = (
    challenger_model_version
    .tags
    .get(
        "quality_gate"
    )
)


registry_eligible = (
    challenger_model_version
    .tags
    .get(
        "registry_eligible"
    )
)


if quality_gate != "PASSED":

    print()
    print(
        "PROMOTION BLOCKED"
    )

    print()

    print(
        "Challenger does not have:"
    )

    print(
        "quality_gate = PASSED"
    )

    sys.exit(1)


if (
    str(
        registry_eligible
    ).lower()
    != "true"
):

    print()
    print(
        "PROMOTION BLOCKED"
    )

    print()

    print(
        "Challenger does not have:"
    )

    print(
        "registry_eligible = true"
    )

    sys.exit(1)


# =========================================================
# 10. PRESERVE CURRENT CHAMPION
#
# Before moving @champion, record the old version as:
#
# previous_champion
#
# This makes rollback easier.
# =========================================================

if current_champion_version is not None:

    client.set_model_version_tag(

        name=(
            REGISTERED_MODEL_NAME
        ),

        version=(
            current_champion_version
        ),

        key=(
            "lifecycle_role"
        ),

        value=(
            "previous_champion"
        ),
    )


# =========================================================
# 11. PROMOTE CHALLENGER
#
# Move:
#
# @champion
#
# to:
#
# challenger version
#
# Application code does not change.
#
# Production still loads:
#
# models:/payflow-payment-success@champion
# =========================================================

client.set_registered_model_alias(

    name=(
        REGISTERED_MODEL_NAME
    ),

    alias=(
        "champion"
    ),

    version=(
        challenger_version
    ),
)


# =========================================================
# 12. TAG NEW CHAMPION
# =========================================================

client.set_model_version_tag(

    name=(
        REGISTERED_MODEL_NAME
    ),

    version=(
        challenger_version
    ),

    key=(
        "lifecycle_role"
    ),

    value=(
        "champion"
    ),
)


# =========================================================
# 13. REMOVE CHALLENGER ALIAS
#
# Once promoted, the version should not be both:
#
# @champion
# @challenger
# =========================================================

client.delete_registered_model_alias(

    name=(
        REGISTERED_MODEL_NAME
    ),

    alias=(
        "challenger"
    ),
)


# =========================================================
# 14. VERIFY PROMOTION
# =========================================================

updated_registered_model = (
    client.get_registered_model(
        REGISTERED_MODEL_NAME
    )
)


updated_aliases = (
    updated_registered_model.aliases
    or {}
)


updated_champion = (
    updated_aliases.get(
        "champion"
    )
)


# =========================================================
# 15. TERMINAL SUMMARY
# =========================================================

print()
print(
    "========================================"
)

print(
    "CHALLENGER PROMOTED SUCCESSFULLY"
)

print(
    "========================================"
)

print(
    f"Registered Model  : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Previous Champion : "
    f"{current_champion_version}"
)

print(
    f"New Champion      : "
    f"{updated_champion}"
)

print()

print(
    "Production URI remains unchanged:"
)

print()

print(
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@champion"
)

print()

print(
    "Application code does not need "
    "a new model version number."
)