import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split transactions chronologically.

    Earlier transactions:
        training

    Middle transactions:
        validation

    Latest transactions:
        testing

    This better resembles production than
    randomly mixing future transactions into training.
    """

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be positive."
        )

    if validation_ratio <= 0:
        raise ValueError(
            "validation_ratio must be positive."
        )

    if (
        train_ratio
        + validation_ratio
        >= 1
    ):
        raise ValueError(
            "Train + validation ratios must be < 1."
        )

    # Sort once by event time.
    ordered = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    total_rows = len(ordered)

    train_end = int(
        total_rows * train_ratio
    )

    validation_end = int(
        total_rows
        * (
            train_ratio
            + validation_ratio
        )
    )

    train_df = ordered.iloc[
        :train_end
    ].copy()

    validation_df = ordered.iloc[
        train_end:validation_end
    ].copy()

    test_df = ordered.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )