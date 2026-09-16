from pathlib import Path

import pandas as pd

def normalize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the original Kaggle column names into
    predictable snake_case names.

    Example:
        "Amount (INR)" -> "amount_inr"
        "Transaction Status" -> "transaction_status"
    """

    df = dataframe.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    return df


def load_transactions(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the raw UPI transaction CSV.

    The raw file is never modified.
    We normalize only the in-memory DataFrame.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    # Read the Kaggle CSV.
    df = pd.read_csv(path)

    # Normalize headers immediately so all downstream
    # code works with consistent column names.
    df = normalize_column_names(df)

    # Convert timestamp into an actual datetime column.
    # Invalid values become NaT so validation can catch them.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    return df