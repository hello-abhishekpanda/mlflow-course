from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np

from dotenv import load_dotenv

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

from payflow.models.thresholding import (
    add_threshold_policy_columns,
    calculate_threshold_metrics,
    create_threshold_metrics_plot,
    get_class_probability,
    select_best_threshold,
    sweep_failure_thresholds,
)


# =========================================================
# 1. LOAD CONFIGURATION
# =========================================================

load_dotenv()


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


DATA_PATH = Path(
    os.getenv(
        "PAYFLOW_DATA_PATH",
        "data/raw/upi_transactions_2024.csv",
    )
)


SELECTION_PATH = Path(
    "artifacts/selected_candidate.json"
)


THRESHOLD_POLICY_PATH = Path(
    "artifacts/threshold_policy.json"
)


THRESHOLD_EXPERIMENT = (
    "payflow-threshold-optimization"
)


# =========================================================
# PRODUCTION POLICY
#
# The candidate may lose no more than this amount of
# overall accuracy relative to the baseline.
#
# Example:
#
# baseline accuracy = 0.95
# max drop          = 0.05
#
# candidate threshold must achieve:
#
# accuracy >= 0.90
#
# IMPORTANT:
#
# Do not keep increasing this value merely to make a model
# pass.
# =========================================================

MAX_ACCURACY_DROP = float(
    os.getenv(
        "PAYFLOW_MAX_ACCURACY_DROP",
        "0.05",
    )
)


# =========================================================
# Minimum required FAILED precision.
#
# We currently require it to be greater than zero.
#
# Later this should come from actual payment business
# economics.
# =========================================================

MINIMUM_FAILED_PRECISION = float(
    os.getenv(
        "PAYFLOW_MIN_FAILED_PRECISION",
        "0.0",
    )
)


# =========================================================
# 2. VALIDATE POLICY CONFIGURATION
# =========================================================

if not 0.0 <= MAX_ACCURACY_DROP <= 1.0:

    raise ValueError(
        "PAYFLOW_MAX_ACCURACY_DROP must be between "
        "0.0 and 1.0."
    )


if not 0.0 <= MINIMUM_FAILED_PRECISION <= 1.0:

    raise ValueError(
        "PAYFLOW_MIN_FAILED_PRECISION must be between "
        "0.0 and 1.0."
    )


# =========================================================
# 3. CONFIGURE MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)


mlflow.set_experiment(
    THRESHOLD_EXPERIMENT
)


print()

print(
    "MLflow Tracking URI:"
)

print(
    mlflow.get_tracking_uri()
)


# =========================================================
# 4. LOAD CANDIDATE SELECTION
#
# Generated earlier by:
#
#     scripts/07_select_candidate.py
#
# It contains BOTH:
#
#     candidate
#     baseline
# =========================================================

if not SELECTION_PATH.exists():

    raise FileNotFoundError(
        "selected_candidate.json was not found.\n\n"
        "Run:\n"
        "python scripts/07_select_candidate.py"
    )


selection = json.loads(
    SELECTION_PATH.read_text(
        encoding="utf-8"
    )
)


candidate = (
    selection[
        "candidate"
    ]
)


baseline = (
    selection[
        "baseline"
    ]
)


candidate_model_uri = (
    candidate[
        "model_uri"
    ]
)


baseline_model_uri = (
    baseline[
        "model_uri"
    ]
)


# =========================================================
# 5. LOAD + VALIDATE DATA
# =========================================================

df = load_transactions(
    DATA_PATH
)


validate_transactions(
    df
)


# ---------------------------------------------------------
# Normalize target strings.
# ---------------------------------------------------------

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.upper()
)


# =========================================================
# 6. ENCODE TARGET
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

    bad_values = (
        df.loc[
            df["target"].isna(),
            TARGET_COLUMN,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    raise ValueError(
        "Could not encode all transaction statuses. "
        f"Unexpected values: {bad_values}"
    )


df["target"] = (
    df["target"]
    .astype(int)
)


# =========================================================
# 7. RECREATE ORIGINAL CHRONOLOGICAL SPLIT
#
# IMPORTANT:
#
# Threshold optimization MUST use VALIDATION data.
#
# It must NOT use the test dataset.
#
# TRAIN
#     ↓
# fit model
#
# VALIDATION
#     ↓
# choose threshold
#
# TEST
#     ↓
# independent final certification
# =========================================================

(
    _,
    validation_df,
    _,
) = chronological_split(
    df
)


X_validation = (
    validation_df[
        FEATURE_COLUMNS
    ]
    .copy()
)


y_validation = (
    validation_df[
        "target"
    ]
    .copy()
)


# =========================================================
# 8. VALIDATE VALIDATION DATASET
# =========================================================

validation_classes = set(
    y_validation.unique()
)


if validation_classes != {0, 1}:

    raise ValueError(
        "Validation set must contain both FAILED and "
        "SUCCESS. "
        f"Found: {sorted(validation_classes)}"
    )


print()

print(
    "Validation Dataset"
)

print(
    "========================================"
)

print(
    f"Rows     : {len(X_validation)}"
)

print(
    f"Features : {len(FEATURE_COLUMNS)}"
)

print()

print(
    "Class distribution:"
)

print(
    y_validation
    .value_counts()
    .sort_index()
)


# =========================================================
# 9. LOAD CANDIDATE MODEL
# =========================================================

print()

print(
    "Loading candidate model..."
)


candidate_model = (
    mlflow_sklearn.load_model(
        candidate_model_uri
    )
)


print(
    "Candidate loaded successfully."
)


# =========================================================
# 10. LOAD BASELINE MODEL
#
# NEW:
#
# Previously we optimized FAILED F1 without considering
# the accuracy policy.
#
# We now need baseline validation accuracy so we can
# establish a minimum acceptable accuracy.
# =========================================================

print()

print(
    "Loading baseline model..."
)


baseline_model = (
    mlflow_sklearn.load_model(
        baseline_model_uri
    )
)


print(
    "Baseline loaded successfully."
)


# =========================================================
# 11. CANDIDATE P(FAILED)
# =========================================================

candidate_failure_probability = (
    get_class_probability(

        model=(
            candidate_model
        ),

        X=(
            X_validation
        ),

        class_label=0,
    )
)


# =========================================================
# 12. BASELINE P(FAILED)
# =========================================================

baseline_failure_probability = (
    get_class_probability(

        model=(
            baseline_model
        ),

        X=(
            X_validation
        ),

        class_label=0,
    )
)


# =========================================================
# 13. CANDIDATE DEFAULT 0.50 METRICS
#
# This shows the candidate's behavior BEFORE threshold
# optimization.
# =========================================================

candidate_default_metrics = (
    calculate_threshold_metrics(

        y_true=(
            y_validation
        ),

        failure_probability=(
            candidate_failure_probability
        ),

        threshold=0.50,
    )
)


# =========================================================
# 14. BASELINE VALIDATION METRICS
#
# Baseline stays at its ordinary 0.50 decision threshold.
#
# Its validation accuracy establishes our accuracy floor.
# =========================================================

baseline_validation_metrics = (
    calculate_threshold_metrics(

        y_true=(
            y_validation
        ),

        failure_probability=(
            baseline_failure_probability
        ),

        threshold=0.50,
    )
)


# =========================================================
# 15. CALCULATE MINIMUM ACCEPTABLE ACCURACY
#
# Example:
#
# baseline accuracy   = 0.950
# allowed loss        = 0.050
#
# minimum accuracy    = 0.900
#
# Any candidate threshold below 0.900 is removed BEFORE
# we maximize FAILED F1.
# =========================================================

minimum_validation_accuracy = max(

    0.0,

    (
        baseline_validation_metrics[
            "accuracy"
        ]
        - MAX_ACCURACY_DROP
    ),
)


# =========================================================
# 16. BUILD THRESHOLD GRID
#
# 199 thresholds:
#
#     0.0100
#     ...
#     0.5000-ish
#     ...
#     0.9900
# =========================================================

thresholds = np.linspace(
    0.01,
    0.99,
    199,
)


# =========================================================
# 17. EVALUATE ALL THRESHOLDS
# =========================================================

threshold_results = (
    sweep_failure_thresholds(

        y_true=(
            y_validation
        ),

        failure_probability=(
            candidate_failure_probability
        ),

        thresholds=(
            thresholds
        ),
    )
)


# =========================================================
# 18. ADD PRODUCTION POLICY COLUMNS
#
# MLflow table will now explicitly show:
#
#     accuracy_policy_pass
#     precision_policy_pass
#     production_eligible
#
# This makes threshold selection auditable.
# =========================================================

threshold_results = (
    add_threshold_policy_columns(

        threshold_results=(
            threshold_results
        ),

        minimum_accuracy=(
            minimum_validation_accuracy
        ),

        minimum_failed_precision=(
            MINIMUM_FAILED_PRECISION
        ),
    )
)


eligible_count = int(
    threshold_results[
        "production_eligible"
    ].sum()
)


print()

print(
    "Threshold Search Policy"
)

print(
    "========================================"
)

print(
    "Baseline Validation Accuracy : "
    f"{baseline_validation_metrics['accuracy']:.6f}"
)

print(
    "Maximum Allowed Accuracy Drop: "
    f"{MAX_ACCURACY_DROP:.6f}"
)

print(
    "Minimum Required Accuracy    : "
    f"{minimum_validation_accuracy:.6f}"
)

print(
    "Minimum FAILED Precision      : "
    f">{MINIMUM_FAILED_PRECISION:.6f}"
)

print(
    "Total Thresholds              : "
    f"{len(threshold_results)}"
)

print(
    "Eligible Thresholds           : "
    f"{eligible_count}"
)


# =========================================================
# 19. SELECT BEST ELIGIBLE THRESHOLD
#
# THIS IS THE MAIN FIX.
#
# OLD:
#
#     maximize FAILED F1 at any cost
#
# NEW:
#
#     meet accuracy policy first
#              ↓
#     meet precision policy
#              ↓
#     maximize FAILED F1
# =========================================================

best_row = (
    select_best_threshold(

        threshold_results=(
            threshold_results
        ),

        minimum_accuracy=(
            minimum_validation_accuracy
        ),

        minimum_failed_precision=(
            MINIMUM_FAILED_PRECISION
        ),
    )
)


selected_threshold = float(
    best_row[
        "threshold"
    ]
)


# =========================================================
# 20. EXTRACT SELECTED VALIDATION METRICS
# =========================================================

selected_metrics = {

    key:
        float(
            best_row[
                key
            ]
        )

    for key in [

        "accuracy",

        "failed_precision",

        "failed_recall",

        "failed_f1",

        "failed_roc_auc",

        "failed_pr_auc",

        "predicted_failure_rate",

        "actual_failure_rate",
    ]
}


# =========================================================
# 21. BUILD THRESHOLD POLICY V2
#
# Policy version changed because the selection logic has
# genuinely changed.
#
# v1:
#     maximize FAILED F1
#
# v2:
#     maximize FAILED F1
#     subject to accuracy + precision constraints
# =========================================================

threshold_policy = {

    "policy_version":
        "v2",

    "candidate_run_name":
        candidate[
            "run_name"
        ],

    "candidate_run_id":
        candidate[
            "run_id"
        ],

    "candidate_model_id":
        candidate[
            "model_id"
        ],

    "candidate_model_uri":
        candidate_model_uri,


    "baseline_run_name":
        baseline[
            "run_name"
        ],

    "baseline_run_id":
        baseline[
            "run_id"
        ],

    "baseline_model_id":
        baseline[
            "model_id"
        ],

    "baseline_model_uri":
        baseline_model_uri,


    "optimized_on":
        "validation",

    "target_event":
        "FAILED",

    "target_label":
        0,


    "objective":
        (
            "maximize_failed_f1_"
            "subject_to_accuracy_policy"
        ),


    "selected_threshold":
        selected_threshold,


    # -----------------------------------------------------
    # Full production constraint definition.
    # -----------------------------------------------------

    "production_constraints": {

        "baseline_validation_accuracy":
            float(
                baseline_validation_metrics[
                    "accuracy"
                ]
            ),

        "max_accuracy_drop":
            float(
                MAX_ACCURACY_DROP
            ),

        "minimum_validation_accuracy":
            float(
                minimum_validation_accuracy
            ),

        "minimum_failed_precision":
            float(
                MINIMUM_FAILED_PRECISION
            ),
    },


    # -----------------------------------------------------
    # Candidate behavior before threshold optimization.
    # -----------------------------------------------------

    "default_threshold_metrics":
        candidate_default_metrics,


    # -----------------------------------------------------
    # Baseline validation reference.
    # -----------------------------------------------------

    "baseline_validation_metrics":
        baseline_validation_metrics,


    # -----------------------------------------------------
    # Final validation metrics at selected threshold.
    # -----------------------------------------------------

    "validation_metrics":
        selected_metrics,


    "threshold_search": {

        "minimum":
            float(
                thresholds.min()
            ),

        "maximum":
            float(
                thresholds.max()
            ),

        "count":
            int(
                len(
                    thresholds
                )
            ),

        "eligible_count":
            int(
                eligible_count
            ),
    },
}


# =========================================================
# 22. SAVE THRESHOLD POLICY LOCALLY
#
# scripts/10_evaluate_thresholded_candidate.py will read
# this file.
# =========================================================

THRESHOLD_POLICY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


THRESHOLD_POLICY_PATH.write_text(

    json.dumps(
        threshold_policy,
        indent=2,
    ),

    encoding="utf-8",
)


# =========================================================
# 23. START MLFLOW THRESHOLD OPTIMIZATION RUN
# =========================================================

with mlflow.start_run(
    run_name="failure-threshold-v2"
) as run:


    # =====================================================
    # TAGS
    # =====================================================

    mlflow.set_tags(
        {
            "project":
                "payflow-ai",

            "domain":
                "upi-payments",

            "problem":
                "payment-success",

            "stage":
                "threshold-optimization",

            "selection_dataset":
                "validation",

            "target_event":
                "FAILED",

            "policy_version":
                "v2",

            "threshold_strategy":
                "constrained",

            "candidate_model_id":
                candidate[
                    "model_id"
                ],

            "baseline_model_id":
                baseline[
                    "model_id"
                ],
        }
    )


    # =====================================================
    # PARAMETERS
    # =====================================================

    mlflow.log_params(
        {
            "candidate_model_id":
                candidate[
                    "model_id"
                ],

            "baseline_model_id":
                baseline[
                    "model_id"
                ],

            "threshold_policy_version":
                "v2",

            "objective":
                (
                    "maximize_failed_f1_"
                    "subject_to_accuracy_policy"
                ),

            "default_threshold":
                0.50,

            "selected_threshold":
                selected_threshold,

            "search_min":
                float(
                    thresholds.min()
                ),

            "search_max":
                float(
                    thresholds.max()
                ),

            "threshold_count":
                int(
                    len(
                        thresholds
                    )
                ),

            "eligible_threshold_count":
                int(
                    eligible_count
                ),

            "max_accuracy_drop":
                float(
                    MAX_ACCURACY_DROP
                ),

            "baseline_validation_accuracy":
                float(
                    baseline_validation_metrics[
                        "accuracy"
                    ]
                ),

            "minimum_validation_accuracy":
                float(
                    minimum_validation_accuracy
                ),

            "minimum_failed_precision":
                float(
                    MINIMUM_FAILED_PRECISION
                ),
        }
    )


    # =====================================================
    # BASELINE VALIDATION METRICS
    # =====================================================

    mlflow.log_metrics(
        {
            f"baseline_validation_{name}":
                float(
                    value
                )

            for (
                name,
                value,
            )
            in baseline_validation_metrics.items()

            if name != "threshold"
        }
    )


    # =====================================================
    # CANDIDATE DEFAULT-THRESHOLD METRICS
    # =====================================================

    mlflow.log_metrics(
        {
            f"default_{name}":
                float(
                    value
                )

            for (
                name,
                value,
            )
            in candidate_default_metrics.items()

            if name != "threshold"
        }
    )


    # =====================================================
    # OPTIMIZED CANDIDATE METRICS
    # =====================================================

    mlflow.log_metrics(
        {
            f"optimized_{name}":
                float(
                    value
                )

            for (
                name,
                value,
            )
            in selected_metrics.items()
        }
    )


    # =====================================================
    # LOG COMPLETE THRESHOLD TABLE
    #
    # Every row contains:
    #
    # threshold
    # accuracy
    # failed precision
    # failed recall
    # failed F1
    # policy PASS/FAIL columns
    # =====================================================

    mlflow.log_table(

        data=(
            threshold_results
        ),

        artifact_file=(
            "threshold/"
            "threshold_sweep_v2.json"
        ),
    )


    # =====================================================
    # CREATE VISUALIZATION
    # =====================================================

    threshold_fig = (
        create_threshold_metrics_plot(

            threshold_results=(
                threshold_results
            ),

            selected_threshold=(
                selected_threshold
            ),

            minimum_accuracy=(
                minimum_validation_accuracy
            ),
        )
    )


    mlflow.log_figure(

        threshold_fig,

        (
            "threshold/"
            "threshold_metrics_v2.png"
        ),
    )


    plt.close(
        threshold_fig
    )


    # =====================================================
    # LOG FROZEN POLICY
    # =====================================================

    mlflow.log_dict(

        threshold_policy,

        (
            "threshold/"
            "threshold_policy_v2.json"
        ),
    )


    # =====================================================
    # TERMINAL SUMMARY
    # =====================================================

    print()

    print(
        "========================================"
    )

    print(
        "CONSTRAINED THRESHOLD OPTIMIZATION"
    )

    print(
        "========================================"
    )

    print(
        f"Candidate Model : "
        f"{candidate['run_name']}"
    )

    print(
        f"Candidate ID    : "
        f"{candidate['model_id']}"
    )

    print()

    print(
        "ACCURACY POLICY"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Baseline Validation Accuracy : "
        f"{baseline_validation_metrics['accuracy']:.6f}"
    )

    print(
        "Maximum Allowed Drop         : "
        f"{MAX_ACCURACY_DROP:.6f}"
    )

    print(
        "Minimum Required Accuracy    : "
        f"{minimum_validation_accuracy:.6f}"
    )

    print(
        "Eligible Thresholds          : "
        f"{eligible_count}"
    )

    print()

    print(
        "DEFAULT POLICY"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Threshold        : "
        "0.500000"
    )

    print(
        "Accuracy         : "
        f"{candidate_default_metrics['accuracy']:.6f}"
    )

    print(
        "FAILED Precision : "
        f"{candidate_default_metrics['failed_precision']:.6f}"
    )

    print(
        "FAILED Recall    : "
        f"{candidate_default_metrics['failed_recall']:.6f}"
    )

    print(
        "FAILED F1        : "
        f"{candidate_default_metrics['failed_f1']:.6f}"
    )

    print()

    print(
        "SELECTED POLICY"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Threshold        : "
        f"{selected_threshold:.6f}"
    )

    print(
        "Accuracy         : "
        f"{selected_metrics['accuracy']:.6f}"
    )

    print(
        "FAILED Precision : "
        f"{selected_metrics['failed_precision']:.6f}"
    )

    print(
        "FAILED Recall    : "
        f"{selected_metrics['failed_recall']:.6f}"
    )

    print(
        "FAILED F1        : "
        f"{selected_metrics['failed_f1']:.6f}"
    )

    print(
        "FAILED PR-AUC    : "
        f"{selected_metrics['failed_pr_auc']:.6f}"
    )

    print(
        "FAILED ROC-AUC   : "
        f"{selected_metrics['failed_roc_auc']:.6f}"
    )

    print()

    print(
        "Policy Version   : v2"
    )

    print(
        "Threshold Policy : "
        f"{THRESHOLD_POLICY_PATH}"
    )

    print(
        f"MLflow Run ID    : "
        f"{run.info.run_id}"
    )

    print(
        "========================================"
    )