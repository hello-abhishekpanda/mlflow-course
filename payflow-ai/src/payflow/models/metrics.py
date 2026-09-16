from typing import Any

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_payment_metrics(
    y_true,
    y_pred,
    success_probability,
) -> dict[str, Any]:
    """
    Calculate metrics for payment success prediction.

    Convention:
        SUCCESS = 1
        FAILED  = 0

    We care specifically about detecting FAILED
    transactions, so we log dedicated failure metrics.
    """

    # Probability that transaction will fail.
    failure_probability = (
        1.0 - success_probability
    )

    # Convert labels so FAILED becomes positive class
    # for PR-AUC calculation.
    y_failed = (
        np.asarray(y_true) == 0
    ).astype(int)

    return {
        # Overall classification accuracy.
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        # Probability-ranking quality for SUCCESS.
        "roc_auc": float(
            roc_auc_score(
                y_true,
                success_probability,
            )
        ),

        # PR-AUC focused specifically on FAILED.
        "failed_pr_auc": float(
            average_precision_score(
                y_failed,
                failure_probability,
            )
        ),

        # When we predict FAILED,
        # how often are we correct?
        "failed_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),

        # Of all real FAILED transactions,
        # how many did we detect?
        "failed_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),

        "failed_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),

        # Penalizes badly calibrated/probabilistic
        # predictions.
        "log_loss": float(
            log_loss(
                y_true,
                success_probability,
            )
        ),
    }