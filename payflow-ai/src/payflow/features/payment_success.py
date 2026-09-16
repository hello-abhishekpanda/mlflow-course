# Target we are trying to predict.
TARGET_COLUMN = "transaction_status"


# Numeric features available at scoring time.
NUMERIC_FEATURES = [
    "amount_inr",
    "hour_of_day",
]


# Categorical features available at scoring time.
CATEGORICAL_FEATURES = [
    "transaction_type",
    "merchant_category",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "day_of_week",
    "is_weekend",
]


# Final model input.
FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)