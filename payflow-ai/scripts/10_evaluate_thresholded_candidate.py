from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn

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

from payflow.models.plots import (
    create_confusion_matrix,
)

from payflow.models.thresholding import (
    calculate_threshold_metrics,
    get_class_probability,
    predict_with_failure_threshold,
)


# =========================================================
# 1. CONFIGURATION
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


FINAL_DECISION_PATH = Path(
    "artifacts/final_gate_decision.json"
)


EXPERIMENT_NAME = (
    "payflow-model-evaluation"
)


# Initial technical policy.
MAX_ACCURACY_DROP = float(
    os.getenv(
        "PAYFLOW_MAX_ACCURACY_DROP",
        "0.05",
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
# 3. LOAD CANDIDATE + THRESHOLD POLICY
# =========================================================

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


candidate = (
    selection["candidate"]
)


baseline = (
    selection["baseline"]
)


# =========================================================
# 4. PROTECT AGAINST STALE THRESHOLD POLICIES
#
# A threshold belongs to one particular model.
# =========================================================

if (
    threshold_policy[
        "candidate_model_id"
    ]
    != candidate[
        "model_id"
    ]
):

    raise RuntimeError(
        "Threshold policy belongs to another model. "
        "Run threshold optimization again."
    )


selected_threshold = float(
    threshold_policy[
        "selected_threshold"
    ]
)


# =========================================================
# 5. LOAD DATASET
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
# 6. USE TEST PARTITION
# =========================================================

(
    _,
    _,
    test_df,
) = chronological_split(
    df
)


X_test = (
    test_df[
        FEATURE_COLUMNS
    ]
    .copy()
)


y_test = (
    test_df["target"]
    .copy()
)


# =========================================================
# 7. LOAD CANDIDATE + BASELINE
# =========================================================

candidate_model = (
    mlflow_sklearn.load_model(
        candidate["model_uri"]
    )
)


baseline_model = (
    mlflow_sklearn.load_model(
        baseline["model_uri"]
    )
)


# =========================================================
# 8. CANDIDATE FAILURE PROBABILITIES
# =========================================================

candidate_failure_probability = (
    get_class_probability(

        candidate_model,

        X_test,

        class_label=0,
    )
)


# =========================================================
# 9. CANDIDATE AT FROZEN THRESHOLD
# =========================================================

candidate_thresholded_metrics = (
    calculate_threshold_metrics(

        y_test,

        candidate_failure_probability,

        selected_threshold,
    )
)


# =========================================================
# 10. SAME MODEL AT DEFAULT 0.50
#
# Lets us prove what threshold tuning actually changed.
# =========================================================

candidate_default_metrics = (
    calculate_threshold_metrics(

        y_test,

        candidate_failure_probability,

        0.50,
    )
)


# =========================================================
# 11. BASELINE
# =========================================================

baseline_failure_probability = (
    get_class_probability(

        baseline_model,

        X_test,

        class_label=0,
    )
)


baseline_metrics = (
    calculate_threshold_metrics(

        y_test,

        baseline_failure_probability,

        0.50,
    )
)


# =========================================================
# 12. FINAL QUALITY GATE
# =========================================================

minimum_accuracy = max(

    0.0,

    baseline_metrics["accuracy"]
    - MAX_ACCURACY_DROP,
)


gate_checks = {

    "failed_f1_improved_over_default": (

        candidate_thresholded_metrics[
            "failed_f1"
        ]
        >
        candidate_default_metrics[
            "failed_f1"
        ]
    ),


    "failed_recall_improved_over_default": (

        candidate_thresholded_metrics[
            "failed_recall"
        ]
        >
        candidate_default_metrics[
            "failed_recall"
        ]
    ),


    "failed_precision_nonzero": (

        candidate_thresholded_metrics[
            "failed_precision"
        ]
        > 0.0
    ),


    "roc_auc_better_than_baseline": (

        candidate_thresholded_metrics[
            "failed_roc_auc"
        ]
        >
        baseline_metrics[
            "failed_roc_auc"
        ]
    ),


    "accuracy_within_policy": (

        candidate_thresholded_metrics[
            "accuracy"
        ]
        >= minimum_accuracy
    ),
}


quality_gate_passed = all(
    gate_checks.values()
)


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

thresholded_predictions = (
    predict_with_failure_threshold(

        candidate_failure_probability,

        selected_threshold,
    )
)


confusion_fig = (
    create_confusion_matrix(
        y_test,
        thresholded_predictions,
    )
)


# =========================================================
# 14. FINAL MLFLOW GOVERNANCE RUN
# =========================================================

with mlflow.start_run(
    run_name="thresholded-production-quality-gate"
) as run:


    mlflow.set_tags(
        {
            "project":
                "payflow-ai",

            "stage":
                "final-production-evaluation",

            "candidate_model_id":
                candidate["model_id"],

            "threshold_policy_version":
                threshold_policy[
                    "policy_version"
                ],

            "quality_gate":
                (
                    "PASSED"
                    if quality_gate_passed
                    else "FAILED"
                ),

            "registry_eligible":
                str(
                    quality_gate_passed
                ).lower(),
        }
    )


    mlflow.log_params(
        {
            "failure_threshold":
                selected_threshold,

            "threshold_optimized_on":
                "validation",

            "threshold_objective":
                threshold_policy[
                    "objective"
                ],

            "max_accuracy_drop":
                MAX_ACCURACY_DROP,
        }
    )


    mlflow.log_metrics(
        {
            f"candidate_thresholded_{name}":
                float(value)

            for (
                name,
                value,
            )
            in candidate_thresholded_metrics.items()

            if name != "threshold"
        }
    )


    mlflow.log_metrics(
        {
            f"candidate_default_{name}":
                float(value)

            for (
                name,
                value,
            )
            in candidate_default_metrics.items()

            if name != "threshold"
        }
    )


    mlflow.log_metrics(
        {
            f"baseline_{name}":
                float(value)

            for (
                name,
                value,
            )
            in baseline_metrics.items()

            if name != "threshold"
        }
    )


    mlflow.log_figure(

        confusion_fig,

        (
            "quality_gate/"
            "thresholded_confusion_matrix.png"
        ),
    )


    plt.close(
        confusion_fig
    )


    # This file becomes the security gate used by our
    # registry script.
    final_decision = {

        "candidate_model_id":
            candidate["model_id"],

        "candidate_model_uri":
            candidate["model_uri"],

        "selected_failure_threshold":
            selected_threshold,

        "threshold_policy_version":
            threshold_policy[
                "policy_version"
            ],

        "candidate_thresholded_metrics":
            candidate_thresholded_metrics,

        "candidate_default_metrics":
            candidate_default_metrics,

        "baseline_metrics":
            baseline_metrics,

        "gate_checks":
            gate_checks,

        "quality_gate_passed":
            quality_gate_passed,

        "registry_eligible":
            quality_gate_passed,
    }


    mlflow.log_dict(

        final_decision,

        (
            "quality_gate/"
            "final_gate_decision.json"
        ),
    )


    FINAL_DECISION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    FINAL_DECISION_PATH.write_text(

        json.dumps(
            final_decision,
            indent=2,
        ),

        encoding="utf-8",
    )


    print()

    print(
        "========================================"
    )

    print(
        "FINAL THRESHOLDED QUALITY GATE"
    )

    print(
        "========================================"
    )


    for (
        name,
        passed,
    ) in gate_checks.items():

        print(
            f"{name:42s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )


    print()


    if quality_gate_passed:

        print(
            "REGISTRY ELIGIBLE: YES"
        )

    else:

        print(
            "REGISTRY ELIGIBLE: NO"
        )