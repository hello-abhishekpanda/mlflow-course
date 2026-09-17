from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# =========================================================
# 1. TRANSACTION INPUT CONTRACT
#
# These are the exact feature columns used by our
# payment-success model.
#
# IMPORTANT:
#
# transaction_status is NOT supplied.
# fraud_flag is NOT supplied.
# transaction_id is NOT supplied.
#
# They are not inference features.
# =========================================================

class TransactionFeatures(
    BaseModel
):
    """
    One PayFlow payment prediction request.
    """

    # Reject unknown fields.
    #
    # This protects us from silently accepting an API
    # payload with spelling mistakes or unsupported fields.
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


    # -----------------------------------------------------
    # Transaction information
    # -----------------------------------------------------

    transaction_type: str = Field(
        min_length=1,
    )


    merchant_category: str = Field(
        min_length=1,
    )


    amount_inr: float = Field(
        ge=0.0,
    )


    # -----------------------------------------------------
    # Customer demographics
    # -----------------------------------------------------

    sender_age_group: str = Field(
        min_length=1,
    )


    receiver_age_group: str = Field(
        min_length=1,
    )


    # -----------------------------------------------------
    # Geography / banking information
    # -----------------------------------------------------

    sender_state: str = Field(
        min_length=1,
    )


    sender_bank: str = Field(
        min_length=1,
    )


    receiver_bank: str = Field(
        min_length=1,
    )


    # -----------------------------------------------------
    # Device / network
    # -----------------------------------------------------

    device_type: str = Field(
        min_length=1,
    )


    network_type: str = Field(
        min_length=1,
    )


    # -----------------------------------------------------
    # Temporal features
    # -----------------------------------------------------

    hour_of_day: int = Field(
        ge=0,
        le=23,
    )


    day_of_week: str = Field(
        min_length=1,
    )


    # Dataset encoding:
    #
    # 0 = weekday
    # 1 = weekend
    is_weekend: int = Field(
        ge=0,
        le=1,
    )


# =========================================================
# 2. SINGLE PREDICTION RESPONSE
# =========================================================

class PredictionResponse(
    BaseModel
):
    """
    Response returned by /v1/predict.
    """

    request_id: str


    prediction: Literal[
        "FAILED",
        "SUCCESS",
    ]


    # Original training label representation.
    prediction_code: Literal[
        0,
        1,
    ]


    failure_probability: float = Field(
        ge=0.0,
        le=1.0,
    )


    success_probability: float = Field(
        ge=0.0,
        le=1.0,
    )


    failure_threshold: float = Field(
        gt=0.0,
        lt=1.0,
    )


    model_name: str

    model_alias: str

    model_version: str

    model_uri: str


    latency_ms: float = Field(
        ge=0.0,
    )


# =========================================================
# 3. BATCH REQUEST
# =========================================================

class BatchPredictionRequest(
    BaseModel
):
    """
    Batch scoring request.

    We intentionally cap batch size so one HTTP request
    cannot consume unlimited API memory.
    """

    transactions: list[
        TransactionFeatures
    ] = Field(
        min_length=1,
        max_length=500,
    )


# =========================================================
# 4. BATCH RESPONSE
# =========================================================

class BatchPredictionResponse(
    BaseModel
):
    """
    Batch scoring response.
    """

    request_id: str

    count: int

    predictions: list[
        PredictionResponse
    ]

    total_latency_ms: float


# =========================================================
# 5. HEALTH RESPONSE
# =========================================================

class HealthResponse(
    BaseModel
):
    """
    Liveness response.

    This only answers:
        Is the API process alive?
    """

    status: Literal[
        "alive"
    ]

    service: str

    api_version: str


# =========================================================
# 6. READINESS RESPONSE
# =========================================================

class ReadinessResponse(
    BaseModel
):
    """
    Readiness means the production model is actually loaded.
    """

    status: Literal[
        "ready"
    ]

    model_name: str

    model_alias: str

    model_version: str


# =========================================================
# 7. MODEL METADATA RESPONSE
# =========================================================

class ModelMetadataResponse(
    BaseModel
):
    """
    Operational metadata for the currently loaded model.
    """

    model_name: str

    model_alias: str

    model_version: str

    model_uri: str

    source_model_id: str | None

    quality_gate: str

    registry_eligible: bool

    failure_threshold: float

    loaded_at_utc: str