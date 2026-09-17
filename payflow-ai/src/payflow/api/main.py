from __future__ import annotations

import logging
import time

from contextlib import (
    asynccontextmanager,
)

from typing import Annotated

from uuid import uuid4

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Response,
    status,
)

from payflow.api.config import (
    load_settings,
)

from payflow.api.model_service import (
    ModelContractError,
    ModelNotReadyError,
    PayFlowModelService,
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
# 1. CONFIGURATION
# =========================================================

settings = (
    load_settings()
)


# =========================================================
# 2. LOGGING
# =========================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)


logger = logging.getLogger(
    "payflow.api"
)


# =========================================================
# 3. MODEL SERVICE
#
# This object exists once for the API process.
# =========================================================

model_service = (
    PayFlowModelService(
        settings
    )
)


# =========================================================
# 4. FASTAPI LIFESPAN
#
# Load the ML model ONCE when the API process starts.
#
# Do NOT download and deserialize the model for every HTTP
# request.
# =========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "Starting PayFlow API."
    )


    try:

        model_service.load()


        metadata = (
            model_service.metadata()
        )


        logger.info(
            "Loaded model %s version=%s alias=@%s "
            "threshold=%.6f",
            metadata[
                "model_name"
            ],
            metadata[
                "model_version"
            ],
            metadata[
                "model_alias"
            ],
            metadata[
                "failure_threshold"
            ],
        )


    except Exception:

        # -------------------------------------------------
        # Keep process alive so /health can report alive
        # while /ready reports 503.
        #
        # Kubernetes/container platforms distinguish
        # liveness from readiness for exactly this reason.
        # -------------------------------------------------

        logger.exception(
            "PayFlow model failed to load."
        )


    # Application starts accepting traffic here.
    yield


    logger.info(
        "Stopping PayFlow API."
    )


# FastAPI recommends lifespan handlers for shared resources
# such as ML models that should be loaded once before the
# application begins serving requests.
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
        "MLflow Model Registry."
    ),

    lifespan=(
        lifespan
    ),
)
# =========================================================
# 5. LIVENESS ENDPOINT
#
# Liveness answers:
#
#     Is the API process running?
#
# It does NOT guarantee the ML model is ready.
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
# 6. READINESS ENDPOINT
#
# Readiness answers:
#
#     Can this API safely accept prediction traffic?
#
# For PayFlow that means:
#
#     @champion resolved
#     quality gate passed
#     registry eligible
#     threshold loaded
#     model loaded
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
# 7. MODEL METADATA
#
# Operational endpoint showing exactly what model this API
# has loaded.
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

            detail="Model is not ready.",
        )


    return ModelMetadataResponse(
        **model_service.metadata()
    )


# =========================================================
# 8. SINGLE TRANSACTION PREDICTION
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

    response: Response,

    x_request_id: Annotated[
        str | None,
        Header(
            alias="X-Request-ID"
        ),
    ] = None,

) -> PredictionResponse:
    """
    Predict whether a single payment is likely to FAILED
    or SUCCESS using the current MLflow @champion model.
    """

    # -----------------------------------------------------
    # Reuse client request ID if supplied.
    #
    # Otherwise create one.
    #
    # Later this becomes useful for distributed tracing.
    # -----------------------------------------------------

    request_id = (

        x_request_id

        or

        str(
            uuid4()
        )
    )


    start = (
        time.perf_counter()
    )


    try:

        # -------------------------------------------------
        # model_dump() converts validated Pydantic input
        # into normal Python dictionary.
        # -------------------------------------------------

        records = [
            payload.model_dump()
        ]


        result = (
            model_service
            .predict_records(
                records
            )[0]
        )


    except ModelNotReadyError as error:

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
            "Prediction failed. request_id=%s",
            request_id,
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


    latency_ms = (

        (
            time.perf_counter()
            - start
        )

        * 1000.0
    )


    metadata = (
        model_service.metadata()
    )


    # Echo the request ID in the HTTP response header too.
    response.headers[
        "X-Request-ID"
    ] = request_id


    logger.info(
        "prediction request_id=%s "
        "prediction=%s "
        "failure_probability=%.6f "
        "version=%s "
        "latency_ms=%.3f",
        request_id,
        result[
            "prediction"
        ],
        result[
            "failure_probability"
        ],
        metadata[
            "model_version"
        ],
        latency_ms,
    )


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
# 9. BATCH PREDICTION
#
# Useful for:
#
# - replay
# - backfill
# - QA
# - small offline batches
#
# This is NOT intended to replace large-scale Spark/batch
# inference.
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

    x_request_id: Annotated[
        str | None,
        Header(
            alias="X-Request-ID"
        ),
    ] = None,

) -> BatchPredictionResponse:

    request_id = (

        x_request_id

        or

        str(
            uuid4()
        )
    )


    start = (
        time.perf_counter()
    )


    try:

        records = [

            transaction.model_dump()

            for transaction
            in payload.transactions
        ]


        results = (
            model_service
            .predict_records(
                records
            )
        )


    except ModelNotReadyError as error:

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
            "Batch prediction failed. "
            "request_id=%s",
            request_id,
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


    total_latency_ms = (

        (
            time.perf_counter()
            - start
        )

        * 1000.0
    )


    metadata = (
        model_service.metadata()
    )


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

                # Batch-level timing is reported separately.
                latency_ms=0.0,
            )
        )


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