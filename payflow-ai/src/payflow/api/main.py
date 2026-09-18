from __future__ import annotations

import logging
import time

from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)

import mlflow

from payflow.api.config import (
    load_settings,
)

from payflow.api.model_service import (
    ModelContractError,
    ModelNotReadyError,
    PayFlowModelService,
)

from payflow.api.observability import (
    configure_logging,
    configure_mlflow_tracing,
    request_logging_middleware,
)

from payflow.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
    ReadinessResponse,
    TransactionFeatures,
)


# =========================================================
# PAYFLOW AI — FASTAPI APPLICATION
#
# PART 6
#   Production inference API
#
# PART 7
#   Dockerized FastAPI
#   MLflow Registry @champion
#   PostgreSQL + MinIO
#
# PART 8
#   Structured logging
#   OpenTelemetry -> Loki -> Grafana
#   MLflow Tracing
#
#
# Runtime:
#
# Client
#   ↓
# FastAPI
#   │
#   ├── Structured Logs
#   │       ↓
#   │   OpenTelemetry
#   │       ↓
#   │      Loki
#   │       ↓
#   │    Grafana
#   │
#   └── MLflow Traces
#           ↓
#       MLflow :5000
#
# =========================================================


# =========================================================
# 1. CONFIGURATION
# =========================================================

settings = (
    load_settings()
)


# =========================================================
# 2. OBSERVABILITY INITIALIZATION
#
# configure_logging()
#
#   Python logging
#       ↓
#   stdout
#       +
#   OpenTelemetry
#       ↓
#   Loki
#
#
# configure_mlflow_tracing()
#
#   MLflow tracing SDK
#       ↓
#   payflow-production-traces
# =========================================================

configure_logging()

# configure_mlflow_tracing()


# =========================================================
# 3. LOGGER
# =========================================================

logger = logging.getLogger(
    "payflow.api"
)


# =========================================================
# 4. MODEL SERVICE
#
# Create one service object for the entire API process.
#
# The model itself is loaded once during application
# startup through the FastAPI lifespan handler.
# =========================================================

model_service = (
    PayFlowModelService(
        settings
    )
)


# =========================================================
# 5. FASTAPI LIFESPAN
#
# Startup:
#
#   FastAPI starts
#       ↓
#   Resolve @champion
#       ↓
#   Validate governance
#       ↓
#   Load failure threshold
#       ↓
#   Load sklearn Pipeline
#
#
# Shutdown:
#
#   log shutdown event
#
#
# IMPORTANT:
#
# We do NOT download the ML model for every request.
# =========================================================
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "application.starting",
        extra={
            "service":
                "payflow-fastapi",
        },
    )


    # =====================================================
    # CONFIGURE MLFLOW TRACING
    #
    # This intentionally happens during application startup,
    # not during Python import.
    #
    # Local:
    #     http://127.0.0.1:5050
    #
    # Docker:
    #     http://mlflow:5000
    # =====================================================

    try:

        configure_mlflow_tracing()

        logger.info(
            "mlflow.tracing_configured",
            extra={
                "service":
                    "payflow-fastapi",
            },
        )


    except Exception:

        # -------------------------------------------------
        # Tracing failure should be visible, but it should
        # not prevent Python from importing the application.
        #
        # For now we allow the API startup to continue.
        # -------------------------------------------------

        logger.exception(
            "mlflow.tracing_configuration_failed",
            extra={
                "service":
                    "payflow-fastapi",
            },
        )


    try:

        model_service.load()

        metadata = (
            model_service.metadata()
        )

        logger.info(
            "model.loaded",
            extra={
                "service":
                    "payflow-fastapi",

                "model_name":
                    metadata[
                        "model_name"
                    ],

                "model_version":
                    metadata[
                        "model_version"
                    ],

                "model_alias":
                    metadata[
                        "model_alias"
                    ],

                "failure_threshold":
                    metadata[
                        "failure_threshold"
                    ],

                "model_uri":
                    metadata[
                        "model_uri"
                    ],
            },
        )


    except Exception:

        logger.exception(
            "model.load_failed",
            extra={
                "service":
                    "payflow-fastapi",
            },
        )


    yield


    logger.info(
        "application.stopping",
        extra={
            "service":
                "payflow-fastapi",
        },
    )

# =========================================================
# 6. FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title=(
        settings.api_title
    ),

    version=(
        settings.api_version
    ),

    description=(
        "Production-style PayFlow AI API backed by "
        "MLflow Model Registry with structured logging "
        "and MLflow tracing."
    ),

    lifespan=(
        lifespan
    ),
)


# =========================================================
# 7. REQUEST LOGGING MIDDLEWARE
#
# Every request receives:
#
#   request_id
#
# The middleware also emits:
#
#   request.started
#   request.completed
#   request.failed
#
# These logs flow to:
#
# FastAPI
#   ↓
# OpenTelemetry Collector
#   ↓
# Loki
#   ↓
# Grafana
# =========================================================

app.middleware(
    "http"
)(
    request_logging_middleware
)


# =========================================================
# 8. LIVENESS
#
# Answers:
#
#     Is the FastAPI process alive?
#
# It does NOT mean the ML model is ready.
# =========================================================

@app.get(
    "/health",

    response_model=(
        HealthResponse
    ),

    tags=[
        "Operations"
    ],
)
def health() -> HealthResponse:

    return HealthResponse(

        status="alive",

        service=(
            settings.api_title
        ),

        api_version=(
            settings.api_version
        ),
    )


# =========================================================
# 9. READINESS
#
# Answers:
#
#     Can PayFlow safely accept prediction traffic?
#
# Ready means:
#
#   @champion resolved
#   quality gate passed
#   registry eligible
#   threshold loaded
#   model loaded
# =========================================================

@app.get(
    "/ready",

    response_model=(
        ReadinessResponse
    ),

    tags=[
        "Operations"
    ],
)
def ready() -> ReadinessResponse:

    if not model_service.ready:

        raise HTTPException(

            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail={
                "status":
                    "not_ready",

                "reason":
                    model_service.load_error
                    or
                    "Model not loaded.",
            },
        )


    metadata = (
        model_service.metadata()
    )


    return ReadinessResponse(

        status="ready",

        model_name=(
            metadata[
                "model_name"
            ]
        ),

        model_alias=(
            metadata[
                "model_alias"
            ]
        ),

        model_version=(
            metadata[
                "model_version"
            ]
        ),
    )


# =========================================================
# 10. MODEL METADATA
#
# Shows exactly which governed MLflow model this API has
# loaded.
# =========================================================

@app.get(
    "/v1/model",

    response_model=(
        ModelMetadataResponse
    ),

    tags=[
        "Model"
    ],
)
def model_metadata() -> ModelMetadataResponse:

    if not model_service.ready:

        raise HTTPException(

            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail=(
                "Model is not ready."
            ),
        )


    return ModelMetadataResponse(
        **model_service.metadata()
    )


# =========================================================
# 11. SINGLE TRANSACTION PREDICTION
#
# Observability:
#
# HTTP request
#      ↓
# request_logging_middleware
#      ↓
# request_id
#      ↓
# MLflow trace
#      ↓
# payflow.predict
#      ↓
# model_service.predict_records()
#
#
# IMPORTANT:
#
# We intentionally DO NOT send the complete payment
# payload to MLflow tracing.
#
# Only operational metadata is traced.
# =========================================================

@app.post(
    "/v1/predict",

    response_model=(
        PredictionResponse
    ),

    tags=[
        "Prediction"
    ],
)
def predict(

    payload: TransactionFeatures,

    request: Request,

    response: Response,

) -> PredictionResponse:
    """
    Predict whether one UPI payment is likely to be FAILED
    or SUCCESS using the current MLflow @champion model.
    """

    # -----------------------------------------------------
    # The request ID has already been created by our
    # request middleware.
    #
    # If the caller supplied:
    #
    #     X-Request-ID
    #
    # the middleware reused it.
    #
    # Therefore logs and traces now use exactly the same
    # correlation identifier.
    # -----------------------------------------------------

    request_id = (
        request.state.request_id
    )


    start = (
        time.perf_counter()
    )


    try:

        # -------------------------------------------------
        # Convert validated Pydantic input into the record
        # format expected by PayFlowModelService.
        # -------------------------------------------------

        records = [
            payload.model_dump()
        ]


        # -------------------------------------------------
        # Retrieve model metadata before creating trace
        # attributes.
        #
        # This does not reload the model.
        # -------------------------------------------------

        metadata = (
            model_service.metadata()
        )


        # =================================================
        # MLFLOW TRACE
        #
        # Creates:
        #
        # payflow.predict
        #
        # under:
        #
        # payflow-production-traces
        #
        # We log operational metadata only.
        # =================================================

        with mlflow.start_span(
            name="payflow.predict"
        ) as span:

            # ---------------------------------------------
            # Minimal trace input.
            #
            # Do NOT store all payment features here.
            # ---------------------------------------------

            span.set_inputs(
                {
                    "request_id":
                        request_id,

                    "feature_count":
                        len(
                            records[0]
                        ),
                }
            )


            # ---------------------------------------------
            # MLflow / model lineage information.
            # ---------------------------------------------

            span.set_attributes(
                {
                    "service.name":
                        "payflow-fastapi",

                    "payflow.request_id":
                        request_id,

                    "mlflow.model.name":
                        str(
                            metadata[
                                "model_name"
                            ]
                        ),

                    "mlflow.model.version":
                        str(
                            metadata[
                                "model_version"
                            ]
                        ),

                    "mlflow.model.alias":
                        str(
                            metadata[
                                "model_alias"
                            ]
                        ),

                    "payflow.failure_threshold":
                        float(
                            metadata[
                                "failure_threshold"
                            ]
                        ),
                }
            )


            # ---------------------------------------------
            # Actual inference.
            # ---------------------------------------------

            result = (
                model_service
                .predict_records(
                    records
                )[0]
            )


            # ---------------------------------------------
            # Safe operational trace output.
            # ---------------------------------------------

            span.set_outputs(
                {
                    "request_id":
                        request_id,

                    "prediction":
                        result[
                            "prediction"
                        ],

                    "prediction_code":
                        result[
                            "prediction_code"
                        ],
                }
            )


    except ModelNotReadyError as error:

        logger.warning(
            "prediction.model_not_ready",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict",
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail=str(
                error
            ),
        ) from error


    except ModelContractError as error:

        logger.warning(
            "prediction.contract_error",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict",

                "error":
                    str(
                        error
                    ),
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_ENTITY
            ),

            detail=str(
                error
            ),
        ) from error


    except Exception as error:

        logger.exception(
            "prediction.failed",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict",
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),

            detail=(
                "Prediction failed."
            ),
        ) from error


    # =====================================================
    # LATENCY
    # =====================================================

    latency_ms = (

        (
            time.perf_counter()
            - start
        )

        * 1000.0
    )


    # -----------------------------------------------------
    # Echo correlation ID back to the caller.
    # -----------------------------------------------------

    response.headers[
        "X-Request-ID"
    ] = request_id


    # =====================================================
    # STRUCTURED PREDICTION LOG
    #
    # Docker stdout
    #       +
    # OTel Collector
    #       ↓
    # Loki
    #       ↓
    # Grafana
    # =====================================================

    logger.info(
        "prediction.completed",
        extra={
            "request_id":
                request_id,

            "endpoint":
                "/v1/predict",

            "prediction":
                result[
                    "prediction"
                ],

            "prediction_code":
                result[
                    "prediction_code"
                ],

            "failure_probability":
                result[
                    "failure_probability"
                ],

            "model_name":
                metadata[
                    "model_name"
                ],

            "model_version":
                metadata[
                    "model_version"
                ],

            "model_alias":
                metadata[
                    "model_alias"
                ],

            "failure_threshold":
                metadata[
                    "failure_threshold"
                ],

            "latency_ms":
                round(
                    latency_ms,
                    3,
                ),
        },
    )


    # =====================================================
    # API RESPONSE
    # =====================================================

    return PredictionResponse(

        request_id=(
            request_id
        ),

        prediction=(
            result[
                "prediction"
            ]
        ),

        prediction_code=(
            result[
                "prediction_code"
            ]
        ),

        failure_probability=(
            result[
                "failure_probability"
            ]
        ),

        success_probability=(
            result[
                "success_probability"
            ]
        ),

        failure_threshold=(
            metadata[
                "failure_threshold"
            ]
        ),

        model_name=(
            metadata[
                "model_name"
            ]
        ),

        model_alias=(
            metadata[
                "model_alias"
            ]
        ),

        model_version=(
            metadata[
                "model_version"
            ]
        ),

        model_uri=(
            metadata[
                "model_uri"
            ]
        ),

        latency_ms=(
            latency_ms
        ),
    )


# =========================================================
# 12. BATCH PREDICTION
#
# Useful for:
#
#   replay
#   backfill
#   QA
#   small offline batches
#
# This endpoint is NOT intended to replace large-scale
# distributed batch inference.
# =========================================================

@app.post(
    "/v1/predict/batch",

    response_model=(
        BatchPredictionResponse
    ),

    tags=[
        "Prediction"
    ],
)
def predict_batch(

    payload: BatchPredictionRequest,

    request: Request,

    response: Response,

) -> BatchPredictionResponse:

    request_id = (
        request.state.request_id
    )


    start = (
        time.perf_counter()
    )


    try:

        # -------------------------------------------------
        # Convert validated Pydantic objects into normal
        # Python dictionaries.
        # -------------------------------------------------

        records = [

            transaction.model_dump()

            for transaction
            in payload.transactions
        ]


        metadata = (
            model_service.metadata()
        )


        # =================================================
        # BATCH TRACE
        #
        # We record batch size, not individual request
        # payloads.
        # =================================================

        with mlflow.start_span(
            name="payflow.predict_batch"
        ) as span:

            span.set_inputs(
                {
                    "request_id":
                        request_id,

                    "transaction_count":
                        len(
                            records
                        ),
                }
            )


            span.set_attributes(
                {
                    "service.name":
                        "payflow-fastapi",

                    "payflow.request_id":
                        request_id,

                    "mlflow.model.name":
                        str(
                            metadata[
                                "model_name"
                            ]
                        ),

                    "mlflow.model.version":
                        str(
                            metadata[
                                "model_version"
                            ]
                        ),

                    "mlflow.model.alias":
                        str(
                            metadata[
                                "model_alias"
                            ]
                        ),

                    "payflow.failure_threshold":
                        float(
                            metadata[
                                "failure_threshold"
                            ]
                        ),

                    "payflow.batch_size":
                        len(
                            records
                        ),
                }
            )


            results = (
                model_service
                .predict_records(
                    records
                )
            )


            failed_count = sum(

                1

                for result
                in results

                if result[
                    "prediction"
                ]
                == "FAILED"
            )


            success_count = (

                len(
                    results
                )

                - failed_count
            )


            span.set_outputs(
                {
                    "request_id":
                        request_id,

                    "prediction_count":
                        len(
                            results
                        ),

                    "failed_count":
                        failed_count,

                    "success_count":
                        success_count,
                }
            )


    except ModelNotReadyError as error:

        logger.warning(
            "batch_prediction.model_not_ready",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict/batch",
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail=str(
                error
            ),
        ) from error


    except ModelContractError as error:

        logger.warning(
            "batch_prediction.contract_error",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict/batch",

                "error":
                    str(
                        error
                    ),
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_ENTITY
            ),

            detail=str(
                error
            ),
        ) from error


    except Exception as error:

        logger.exception(
            "batch_prediction.failed",
            extra={
                "request_id":
                    request_id,

                "endpoint":
                    "/v1/predict/batch",
            },
        )


        raise HTTPException(

            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),

            detail=(
                "Batch prediction failed."
            ),
        ) from error


    # =====================================================
    # TOTAL BATCH LATENCY
    # =====================================================

    total_latency_ms = (

        (
            time.perf_counter()
            - start
        )

        * 1000.0
    )


    # -----------------------------------------------------
    # Return correlation ID to caller.
    # -----------------------------------------------------

    response.headers[
        "X-Request-ID"
    ] = request_id


    # =====================================================
    # CONVERT MODEL RESULTS INTO API RESPONSES
    # =====================================================

    prediction_responses = []


    for result in results:

        prediction_responses.append(

            PredictionResponse(

                request_id=(
                    request_id
                ),

                prediction=(
                    result[
                        "prediction"
                    ]
                ),

                prediction_code=(
                    result[
                        "prediction_code"
                    ]
                ),

                failure_probability=(
                    result[
                        "failure_probability"
                    ]
                ),

                success_probability=(
                    result[
                        "success_probability"
                    ]
                ),

                failure_threshold=(
                    metadata[
                        "failure_threshold"
                    ]
                ),

                model_name=(
                    metadata[
                        "model_name"
                    ]
                ),

                model_alias=(
                    metadata[
                        "model_alias"
                    ]
                ),

                model_version=(
                    metadata[
                        "model_version"
                    ]
                ),

                model_uri=(
                    metadata[
                        "model_uri"
                    ]
                ),

                # -----------------------------------------
                # Batch timing is returned at batch level,
                # not duplicated as fake per-row latency.
                # -----------------------------------------

                latency_ms=0.0,
            )
        )


    # =====================================================
    # STRUCTURED BATCH LOG
    # =====================================================

    logger.info(
        "batch_prediction.completed",
        extra={
            "request_id":
                request_id,

            "endpoint":
                "/v1/predict/batch",

            "transaction_count":
                len(
                    prediction_responses
                ),

            "failed_count":
                failed_count,

            "success_count":
                success_count,

            "model_name":
                metadata[
                    "model_name"
                ],

            "model_version":
                metadata[
                    "model_version"
                ],

            "model_alias":
                metadata[
                    "model_alias"
                ],

            "failure_threshold":
                metadata[
                    "failure_threshold"
                ],

            "latency_ms":
                round(
                    total_latency_ms,
                    3,
                ),
        },
    )


    # =====================================================
    # BATCH RESPONSE
    # =====================================================

    return BatchPredictionResponse(

        request_id=(
            request_id
        ),

        count=(
            len(
                prediction_responses
            )
        ),

        predictions=(
            prediction_responses
        ),

        total_latency_ms=(
            total_latency_ms
        ),
    )