from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# PAYFLOW THRESHOLDING UTILITIES
#
# PayFlow label convention:
#
#     FAILED  = 0
#     SUCCESS = 1
#
# IMPORTANT:
#
# The ML model estimates probabilities.
#
# The threshold decides how those probabilities are
# converted into business decisions.
#
# These are two different concerns:
#
# Model:
#     P(FAILED), P(SUCCESS)
#
# Decision Policy:
#     if P(FAILED) >= threshold:
#         FAILED
#     else:
#         SUCCESS
# =========================================================


# =========================================================
# 1. GET PROBABILITY FOR ONE CLASS
# =========================================================

def get_class_probability(
    model: Any,
    X: pd.DataFrame,
    class_label: int,
) -> np.ndarray:
    """
    Return the predicted probability for one class.

    Example:

        class_label = 0

    means:

        return P(FAILED)

    We intentionally inspect classifier.classes_ instead
    of blindly assuming predict_proba()[:, 0] or [:, 1].

    This keeps the function safe even if sklearn's class
    ordering is different from what we expect.
    """

    # -----------------------------------------------------
    # Complete probability matrix.
    #
    # Example:
    #
    # [
    #   [0.12, 0.88],
    #   [0.65, 0.35],
    #   ...
    # ]
    # -----------------------------------------------------

    probability_matrix = (
        model.predict_proba(
            X
        )
    )


    # -----------------------------------------------------
    # Our logged MLflow model is an sklearn Pipeline:
    #
    #     preprocessor
    #         ↓
    #     classifier
    #
    # Access the fitted classifier.
    # -----------------------------------------------------

    if not hasattr(
        model,
        "named_steps",
    ):
        raise TypeError(
            "Expected an sklearn Pipeline with "
            "'named_steps'."
        )


    if "classifier" not in model.named_steps:
        raise KeyError(
            "The sklearn Pipeline does not contain "
            "a 'classifier' step."
        )


    classifier = (
        model
        .named_steps[
            "classifier"
        ]
    )


    # -----------------------------------------------------
    # classes_ tells us the exact predict_proba() ordering.
    # -----------------------------------------------------

    if not hasattr(
        classifier,
        "classes_",
    ):
        raise AttributeError(
            "Classifier does not expose classes_."
        )


    classes = list(
        classifier.classes_
    )


    if class_label not in classes:
        raise ValueError(
            f"Class {class_label} was not found. "
            f"Available classes: {classes}"
        )


    class_index = (
        classes.index(
            class_label
        )
    )


    return np.asarray(
        probability_matrix[
            :,
            class_index,
        ],
        dtype=float,
    )


# =========================================================
# 2. APPLY A FAILURE THRESHOLD
# =========================================================

def predict_with_failure_threshold(
    failure_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert P(FAILED) into PayFlow business predictions.

    PayFlow labels:

        FAILED  = 0
        SUCCESS = 1

    Decision rule:

        if P(FAILED) >= threshold:
            predict FAILED
        else:
            predict SUCCESS
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be strictly between "
            "0.0 and 1.0."
        )


    failure_probability = np.asarray(
        failure_probability,
        dtype=float,
    )


    predictions = np.where(

        failure_probability
        >= threshold,

        # FAILED
        0,

        # SUCCESS
        1,
    )


    return predictions.astype(
        int
    )


# =========================================================
# 3. CALCULATE METRICS FOR ONE THRESHOLD
# =========================================================

def calculate_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    failure_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Calculate PayFlow model metrics for one decision
    threshold.

    FAILED=0 is the business event we care about.

    Threshold-dependent metrics:

        accuracy
        failed_precision
        failed_recall
        failed_f1
        predicted_failure_rate

    Threshold-independent probability-ranking metrics:

        failed_roc_auc
        failed_pr_auc
    """

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    )


    failure_probability = np.asarray(
        failure_probability,
        dtype=float,
    )


    # -----------------------------------------------------
    # Defensive validation.
    # -----------------------------------------------------

    if (
        len(y_true_array)
        != len(failure_probability)
    ):
        raise ValueError(
            "y_true and failure_probability must contain "
            "the same number of rows."
        )


    if len(y_true_array) == 0:
        raise ValueError(
            "Cannot calculate threshold metrics "
            "for an empty dataset."
        )


    unique_classes = set(
        np.unique(
            y_true_array
        ).tolist()
    )


    if unique_classes != {0, 1}:
        raise ValueError(
            "Expected binary labels {0, 1}. "
            f"Found: {sorted(unique_classes)}"
        )


    # -----------------------------------------------------
    # Convert probabilities into business predictions.
    # -----------------------------------------------------

    predictions = (
        predict_with_failure_threshold(

            failure_probability=(
                failure_probability
            ),

            threshold=(
                threshold
            ),
        )
    )


    # =====================================================
    # OVERALL ACCURACY
    # =====================================================

    accuracy = (
        accuracy_score(
            y_true_array,
            predictions,
        )
    )


    # =====================================================
    # FAILED PRECISION
    #
    # Of everything we predicted FAILED,
    # how much was truly FAILED?
    # =====================================================

    failed_precision = (
        precision_score(

            y_true_array,

            predictions,

            pos_label=0,

            zero_division=0,
        )
    )


    # =====================================================
    # FAILED RECALL
    #
    # Of every real FAILED transaction,
    # how many did the model detect?
    # =====================================================

    failed_recall = (
        recall_score(

            y_true_array,

            predictions,

            pos_label=0,

            zero_division=0,
        )
    )


    # =====================================================
    # FAILED F1
    #
    # Harmonic mean of FAILED precision and recall.
    # =====================================================

    failed_f1 = (
        f1_score(

            y_true_array,

            predictions,

            pos_label=0,

            zero_division=0,
        )
    )


    # =====================================================
    # PROBABILITY RANKING METRICS
    #
    # sklearn's ranking metrics expect positive event = 1.
    #
    # Our source labels are:
    #
    #     FAILED  = 0
    #     SUCCESS = 1
    #
    # Therefore create a temporary binary representation:
    #
    #     FAILED  -> 1
    #     SUCCESS -> 0
    # =====================================================

    failure_target = (
        y_true_array == 0
    ).astype(
        int
    )


    failed_roc_auc = (
        roc_auc_score(

            failure_target,

            failure_probability,
        )
    )


    failed_pr_auc = (
        average_precision_score(

            failure_target,

            failure_probability,
        )
    )


    # =====================================================
    # PREDICTED FAILURE RATE
    #
    # Very useful when tuning thresholds.
    #
    # An extremely low threshold may classify almost every
    # transaction as FAILED.
    #
    # That could improve recall while destroying business
    # usefulness and accuracy.
    # =====================================================

    predicted_failure_rate = float(
        np.mean(
            predictions == 0
        )
    )


    actual_failure_rate = float(
        np.mean(
            y_true_array == 0
        )
    )


    return {

        "threshold":
            float(threshold),

        "accuracy":
            float(accuracy),

        "failed_precision":
            float(failed_precision),

        "failed_recall":
            float(failed_recall),

        "failed_f1":
            float(failed_f1),

        "failed_roc_auc":
            float(failed_roc_auc),

        "failed_pr_auc":
            float(failed_pr_auc),

        "predicted_failure_rate":
            float(
                predicted_failure_rate
            ),

        "actual_failure_rate":
            float(
                actual_failure_rate
            ),
    }


# =========================================================
# 4. SWEEP ACROSS MANY THRESHOLDS
# =========================================================

def sweep_failure_thresholds(
    y_true: pd.Series | np.ndarray,
    failure_probability: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate every candidate threshold.

    Example:

        0.01
        0.015
        0.02
        ...
        0.99

    One output row is created for every threshold.
    """

    results: list[
        dict[str, float]
    ] = []


    for threshold in thresholds:

        threshold_value = float(
            threshold
        )


        result = (
            calculate_threshold_metrics(

                y_true=(
                    y_true
                ),

                failure_probability=(
                    failure_probability
                ),

                threshold=(
                    threshold_value
                ),
            )
        )


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    if results_df.empty:
        raise RuntimeError(
            "Threshold sweep produced no results."
        )


    return results_df


# =========================================================
# 5. SELECT BEST PRODUCTION-ELIGIBLE THRESHOLD
# =========================================================

def select_best_threshold(
    threshold_results: pd.DataFrame,
    minimum_accuracy: float,
    minimum_failed_precision: float = 0.0,
) -> pd.Series:
    """
    Select the best validation threshold while respecting
    production constraints.

    FIRST:

        reject thresholds that violate accuracy policy

    SECOND:

        reject thresholds with insufficient FAILED
        precision

    THIRD:

        among eligible thresholds maximize FAILED F1

    Tie breakers:

        1. FAILED recall
        2. FAILED precision
        3. accuracy
        4. higher threshold

    IMPORTANT:

        This must run against VALIDATION data only.

        Never optimize threshold using holdout test data.
    """

    required_columns = {
        "threshold",
        "accuracy",
        "failed_precision",
        "failed_recall",
        "failed_f1",
    }


    missing_columns = (
        required_columns
        - set(
            threshold_results.columns
        )
    )


    if missing_columns:
        raise ValueError(
            "Threshold results are missing required "
            f"columns: {sorted(missing_columns)}"
        )


    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError(
            "minimum_accuracy must be between 0 and 1."
        )


    if not 0.0 <= minimum_failed_precision <= 1.0:
        raise ValueError(
            "minimum_failed_precision must be between "
            "0 and 1."
        )


    # =====================================================
    # APPLY PRODUCTION CONSTRAINTS
    # =====================================================

    eligible_thresholds = (
        threshold_results[
            (
                threshold_results[
                    "accuracy"
                ]
                >= minimum_accuracy
            )
            &
            (
                threshold_results[
                    "failed_precision"
                ]
                > minimum_failed_precision
            )
        ]
        .copy()
    )


    # =====================================================
    # FAIL CLOSED
    #
    # If no threshold can satisfy production constraints,
    # do NOT silently lower standards.
    #
    # The underlying model requires improvement.
    # =====================================================

    if eligible_thresholds.empty:

        best_accuracy = float(
            threshold_results[
                "accuracy"
            ].max()
        )


        best_failed_f1 = float(
            threshold_results[
                "failed_f1"
            ].max()
        )


        best_failed_recall = float(
            threshold_results[
                "failed_recall"
            ].max()
        )


        raise RuntimeError(
            "\n"
            "NO VALIDATION THRESHOLD SATISFIES THE "
            "PRODUCTION POLICY.\n"
            "\n"
            f"Minimum required accuracy : "
            f"{minimum_accuracy:.6f}\n"
            f"Best observed accuracy    : "
            f"{best_accuracy:.6f}\n"
            f"Best observed FAILED F1   : "
            f"{best_failed_f1:.6f}\n"
            f"Best observed FAILED Recall: "
            f"{best_failed_recall:.6f}\n"
            "\n"
            "Do not weaken the production gate merely to "
            "register the model. The model or features "
            "should be improved."
        )


    # =====================================================
    # RANK ONLY ELIGIBLE THRESHOLDS
    # =====================================================

    ranked = (
        eligible_thresholds
        .sort_values(

            by=[
                "failed_f1",
                "failed_recall",
                "failed_precision",
                "accuracy",
                "threshold",
            ],

            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    return ranked.iloc[0]


# =========================================================
# 6. ADD POLICY FLAGS TO THRESHOLD TABLE
# =========================================================

def add_threshold_policy_columns(
    threshold_results: pd.DataFrame,
    minimum_accuracy: float,
    minimum_failed_precision: float = 0.0,
) -> pd.DataFrame:
    """
    Add explicit PASS/FAIL policy columns to the threshold
    sweep table.

    This makes the MLflow artifact easier to inspect.

    New columns:

        accuracy_policy_pass
        precision_policy_pass
        production_eligible
    """

    result = (
        threshold_results
        .copy()
    )


    result[
        "accuracy_policy_pass"
    ] = (
        result["accuracy"]
        >= minimum_accuracy
    )


    result[
        "precision_policy_pass"
    ] = (
        result["failed_precision"]
        > minimum_failed_precision
    )


    result[
        "production_eligible"
    ] = (
        result[
            "accuracy_policy_pass"
        ]
        &
        result[
            "precision_policy_pass"
        ]
    )


    return result


# =========================================================
# 7. THRESHOLD METRICS VISUALIZATION
# =========================================================

def create_threshold_metrics_plot(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
    minimum_accuracy: float | None = None,
):
    """
    Plot threshold vs:

        FAILED Precision
        FAILED Recall
        FAILED F1
        Accuracy

    Also show:

        selected threshold
        optional minimum accuracy policy
    """

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )


    # -----------------------------------------------------
    # FAILED precision.
    # -----------------------------------------------------

    ax.plot(

        threshold_results[
            "threshold"
        ],

        threshold_results[
            "failed_precision"
        ],

        label=(
            "FAILED Precision"
        ),
    )


    # -----------------------------------------------------
    # FAILED recall.
    # -----------------------------------------------------

    ax.plot(

        threshold_results[
            "threshold"
        ],

        threshold_results[
            "failed_recall"
        ],

        label=(
            "FAILED Recall"
        ),
    )


    # -----------------------------------------------------
    # FAILED F1.
    # -----------------------------------------------------

    ax.plot(

        threshold_results[
            "threshold"
        ],

        threshold_results[
            "failed_f1"
        ],

        label=(
            "FAILED F1"
        ),
    )


    # -----------------------------------------------------
    # Overall accuracy.
    #
    # This is important now because accuracy is part of
    # our threshold-selection constraint.
    # -----------------------------------------------------

    ax.plot(

        threshold_results[
            "threshold"
        ],

        threshold_results[
            "accuracy"
        ],

        label=(
            "Accuracy"
        ),
    )


    # =====================================================
    # SELECTED THRESHOLD
    # =====================================================

    ax.axvline(

        selected_threshold,

        linestyle="--",

        label=(
            "Selected Threshold "
            f"{selected_threshold:.4f}"
        ),
    )


    # =====================================================
    # ACCURACY POLICY FLOOR
    # =====================================================

    if minimum_accuracy is not None:

        ax.axhline(

            minimum_accuracy,

            linestyle=":",

            label=(
                "Minimum Accuracy "
                f"{minimum_accuracy:.4f}"
            ),
        )


    ax.set_title(
        "PayFlow Constrained FAILED Threshold Optimization"
    )


    ax.set_xlabel(
        "P(FAILED) Decision Threshold"
    )


    ax.set_ylabel(
        "Metric Value"
    )


    ax.set_ylim(
        0.0,
        1.0,
    )


    ax.legend()


    fig.tight_layout()


    return fig