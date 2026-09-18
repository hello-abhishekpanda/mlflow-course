from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import mlflow

from fastapi import Request

from opentelemetry._logs import (
    set_logger_provider,
)

from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)

from opentelemetry.instrumentation.logging.handler import (
    LoggingHandler,
)

from opentelemetry.sdk._logs import (
    LoggerProvider,
)

from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
)

from opentelemetry.sdk.resources import (
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)


# =========================================================
# PAYFLOW AI — OBSERVABILITY
#
# Runtime observability:
#
# Logs:
#
#   Python logging
#        ↓
#   OpenTelemetry
#        ↓
#   OTel Collector
#        ↓
#     Loki
#        ↓
#    Grafana
#
# Traces:
#
#   MLflow Tracing SDK
#        ↓
#   MLflow Tracking Server
#        ↓
#   MLflow Traces UI
# =========================================================


SERVICE = os.getenv(
    "OTEL_SERVICE_NAME",
    "payflow-fastapi",
)

ENVIRONMENT = os.getenv(
    "PAYFLOW_ENVIRONMENT",
    "local-docker",
)

TRACE_EXPERIMENT = os.getenv(
    "PAYFLOW_TRACE_EXPERIMENT",
    "payflow-production-traces",
)

OTEL_LOGS_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "http://otel-collector:4318/v1/logs",
)


# =========================================================
# JSON CONSOLE FORMATTER
#
# Docker still receives readable structured logs even if
# Loki or the Collector is temporarily unavailable.
# =========================================================


class JsonFormatter(
    logging.Formatter
):
    """
    Convert Python log records into structured JSON.

    Important fields are kept flat so Docker logs and
    humans can read them easily.
    """

    RESERVED_FIELDS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }


    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload: dict[str, Any] = {

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "level":
                record.levelname,

            "logger":
                record.name,

            "message":
                record.getMessage(),

            "service":
                SERVICE,

            "environment":
                ENVIRONMENT,
        }


        for key, value in record.__dict__.items():

            if (
                key not in self.RESERVED_FIELDS
                and
                not key.startswith("_")
            ):

                payload[key] = value


        if record.exc_info:

            payload[
                "exception"
            ] = self.formatException(
                record.exc_info
            )


        return json.dumps(
            payload,
            default=str,
        )


# =========================================================
# LOGGING CONFIGURATION
# =========================================================


def configure_logging() -> None:
    """
    Configure two log destinations:

    1. stdout
       Docker can always show the application log.

    2. OTLP
       OpenTelemetry Collector -> Loki -> Grafana.
    """

    root = logging.getLogger()

    root.setLevel(
        logging.INFO
    )

    root.handlers.clear()


    # -----------------------------------------------------
    # Console / Docker handler
    # -----------------------------------------------------

    console_handler = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    console_handler.setFormatter(
        JsonFormatter()
    )

    root.addHandler(
        console_handler
    )


    # -----------------------------------------------------
    # OpenTelemetry resource
    # -----------------------------------------------------

    resource = Resource.create(
        {
            SERVICE_NAME:
                SERVICE,

            SERVICE_VERSION:
                "1.0.0",

            "deployment.environment":
                ENVIRONMENT,

            "service.namespace":
                "payflow-ai",
        }
    )


    logger_provider = (
        LoggerProvider(
            resource=resource
        )
    )


    exporter = (
        OTLPLogExporter(
            endpoint=(
                OTEL_LOGS_ENDPOINT
            )
        )
    )


    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            exporter
        )
    )


    set_logger_provider(
        logger_provider
    )


    otel_handler = LoggingHandler(
        level=logging.INFO,
        logger_provider=(
            logger_provider
        ),
    )


    root.addHandler(
        otel_handler
    )


# =========================================================
# MLFLOW TRACING
# =========================================================


def configure_mlflow_tracing() -> None:
    """
    Configure MLflow tracing for the running environment.

    Local host:
        http://127.0.0.1:5050

    Docker:
        MLFLOW_TRACKING_URI=http://mlflow:5000

    IMPORTANT:
    This function should be called during FastAPI startup,
    not while importing main.py.
    """

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5050",
    )

    trace_experiment = os.getenv(
        "PAYFLOW_TRACE_EXPERIMENT",
        "payflow-production-traces",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_registry_uri(
        tracking_uri
    )

    # -----------------------------------------------------
    # Create / select the tracing experiment.
    #
    # This requires the MLflow server to actually be alive,
    # which is why it belongs in FastAPI startup.
    # -----------------------------------------------------

    mlflow.set_experiment(
        trace_experiment
    )

# =========================================================
# REQUEST ID
# =========================================================


def get_request_id(
    request: Request,
) -> str:
    """
    Reuse a caller-provided correlation ID when available.
    Otherwise create one.
    """

    existing = request.headers.get(
        "X-Request-ID"
    )

    if existing:

        return existing


    return str(
        uuid.uuid4()
    )


# =========================================================
# REQUEST LOGGING MIDDLEWARE
# =========================================================


async def request_logging_middleware(
    request: Request,
    call_next,
):
    """
    Log request start/end without logging the complete
    payment payload.

    We deliberately avoid logging all inference features.
    """

    logger = logging.getLogger(
        "payflow.api.request"
    )


    request_id = get_request_id(
        request
    )


    request.state.request_id = (
        request_id
    )


    logger.info(
        "request.started",
        extra={
            "request_id":
                request_id,

            "http_method":
                request.method,

            "http_path":
                request.url.path,
        },
    )


    try:

        response = await call_next(
            request
        )


    except Exception:

        logger.exception(
            "request.failed",
            extra={
                "request_id":
                    request_id,

                "http_method":
                    request.method,

                "http_path":
                    request.url.path,
            },
        )

        raise


    response.headers[
        "X-Request-ID"
    ] = request_id


    logger.info(
        "request.completed",
        extra={
            "request_id":
                request_id,

            "http_method":
                request.method,

            "http_path":
                request.url.path,

            "status_code":
                response.status_code,
        },
    )


    return response