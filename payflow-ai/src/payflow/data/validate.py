import pandas as pd


# These are the 17 logical fields we expect
# from the UPI Transactions 2024 dataset.
REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "transaction_type",
    "merchant_category",
    "amount_inr",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "fraud_flag",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
}


def validate_transactions(
    df: pd.DataFrame,
) -> None:
    """
    Fail early if the input dataset does not satisfy
    our minimum PayFlow AI data contract.
    """

    # -------------------------------------------------
    # 1. Required columns
    # -------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------
    # 2. Transaction ID
    # -------------------------------------------------

    if df["transaction_id"].isna().any():
        raise ValueError(
            "transaction_id contains null values."
        )

    # -------------------------------------------------
    # 3. Timestamp
    # -------------------------------------------------

    if df["timestamp"].isna().any():
        raise ValueError(
            "One or more timestamps could not be parsed."
        )

    # -------------------------------------------------
    # 4. INR amount
    # -------------------------------------------------

    if df["amount_inr"].isna().any():
        raise ValueError(
            "amount_inr contains null values."
        )

    if not df["amount_inr"].ge(0).all():
        raise ValueError(
            "Negative INR transaction amount found."
        )

    # -------------------------------------------------
    # 5. Hour
    # -------------------------------------------------

    if not df["hour_of_day"].between(
        0,
        23,
    ).all():
        raise ValueError(
            "hour_of_day must be between 0 and 23."
        )

    # -------------------------------------------------
    # 6. Payment outcome
    # -------------------------------------------------

    valid_statuses = {
        "SUCCESS",
        "FAILED",
    }

    actual_statuses = set(
        df["transaction_status"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if not actual_statuses.issubset(
        valid_statuses
    ):
        raise ValueError(
            "Unexpected transaction_status values: "
            f"{actual_statuses}"
        )

    # -------------------------------------------------
    # 7. Fraud flag
    # -------------------------------------------------

    fraud_values = set(
        pd.to_numeric(
            df["fraud_flag"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if not fraud_values.issubset({0, 1}):
        raise ValueError(
            "fraud_flag must contain only 0 and 1."
        )