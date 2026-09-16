import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def create_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Build confusion matrix figure.

    Label order:
        0 = FAILED
        1 = SUCCESS
    """

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "FAILED",
            "SUCCESS",
        ],
        ax=ax,
    )

    ax.set_title(
        "Payment Success — Confusion Matrix"
    )

    fig.tight_layout()

    return fig


def create_roc_curve(
    y_true,
    success_probability,
):
    """
    Build ROC curve for SUCCESS probability.
    """

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        success_probability,
        ax=ax,
    )

    ax.set_title(
        "Payment Success — ROC Curve"
    )

    fig.tight_layout()

    return fig


def create_failure_pr_curve(
    y_true,
    success_probability,
):
    """
    Build Precision-Recall curve with FAILED
    treated as the positive event.
    """

    y_failed = (
        y_true == 0
    ).astype(int)

    failure_probability = (
        1.0 - success_probability
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_failed,
        failure_probability,
        ax=ax,
    )

    ax.set_title(
        "Payment Failure — Precision Recall"
    )

    fig.tight_layout()

    return fig