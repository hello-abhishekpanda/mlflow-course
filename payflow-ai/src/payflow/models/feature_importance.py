from __future__ import annotations

import pandas as pd

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def extract_feature_importance(
    model_pipeline: Pipeline,
) -> pd.DataFrame | None:
    """
    Extract feature importance from a trained PayFlow pipeline.

    Works for classifiers that expose:

        feature_importances_

    Examples:
        RandomForestClassifier
        XGBClassifier

    Returns:
        DataFrame with:
            feature
            importance

        sorted from most important to least important.

    Returns None when the classifier does not expose
    feature importance.
    """

    # -----------------------------------------------------
    # Get the fitted preprocessing component.
    # -----------------------------------------------------

    preprocessor = (
        model_pipeline
        .named_steps["preprocessor"]
    )

    # -----------------------------------------------------
    # Get the fitted classifier.
    # -----------------------------------------------------

    classifier = (
        model_pipeline
        .named_steps["classifier"]
    )

    # -----------------------------------------------------
    # Logistic Regression and some other estimators do not
    # expose feature_importances_.
    # -----------------------------------------------------

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        return None

    # -----------------------------------------------------
    # After OneHotEncoder, one raw categorical feature may
    # become many model features.
    #
    # Example:
    #
    # sender_bank
    #
    # becomes:
    #
    # categorical__sender_bank_HDFC
    # categorical__sender_bank_SBI
    # categorical__sender_bank_ICICI
    # ...
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        classifier.feature_importances_
    )

    # -----------------------------------------------------
    # Safety check:
    # feature-name count must match importance count.
    # -----------------------------------------------------

    if (
        len(feature_names)
        != len(importances)
    ):
        raise ValueError(
            "Feature-name count does not match "
            "feature-importance count."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result

def create_feature_importance_plot(
    importance_df: pd.DataFrame,
    top_n: int = 20,
):
    """
    Plot the top N transformed features.

    We intentionally display only the strongest features
    because one-hot encoding can create many columns.
    """

    top_features = (
        importance_df
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        top_features["feature"],
        top_features["importance"],
    )

    ax.set_title(
        f"Top {top_n} Feature Importances"
    )

    ax.set_xlabel(
        "Importance"
    )

    fig.tight_layout()

    return fig