from __future__ import annotations

import json
import os
import time

from pathlib import Path
from typing import Any

import httpx

from dotenv import load_dotenv

from payflow.data.load import (
    load_transactions,
)

from payflow.features.payment_success import (
    FEATURE_COLUMNS,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()


DATA_PATH = Path(
    os.getenv(
        "PAYFLOW_DATA_PATH",
        "data/raw/upi_transactions_2024.csv",
    )
)


API_URL = os.getenv(
    "PAYFLOW_API_URL",
    "http://127.0.0.1:8001",
)


# ---------------------------------------------------------
# The FastAPI process loads the MLflow model during startup.
#
# On a cold start this can take several seconds because
# MLflow may need to download model artifacts.
#
# Therefore our smoke test will wait for /health rather
# than immediately failing.
# ---------------------------------------------------------

API_STARTUP_TIMEOUT_SECONDS = float(
    os.getenv(
        "PAYFLOW_API_STARTUP_TIMEOUT_SECONDS",
        "90",
    )
)


API_STARTUP_POLL_SECONDS = float(
    os.getenv(
        "PAYFLOW_API_STARTUP_POLL_SECONDS",
        "2",
    )
)


# =========================================================
# 2. HTTP TIMEOUT CONFIGURATION
# =========================================================

HTTP_TIMEOUT = httpx.Timeout(

    # Time allowed to establish the TCP connection.
    connect=5.0,

    # Time allowed while waiting for response data.
    read=30.0,

    # Time allowed while sending request data.
    write=30.0,

    # Time allowed while waiting for a pooled connection.
    pool=5.0,
)


# =========================================================
# 3. PRINT JSON HELPER
# =========================================================

def print_json(
    title: str,
    value: Any,
) -> None:
    """
    Print JSON in a consistent format for our Part 6
    terminal screenshots.
    """

    print()

    print(
        title
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            value,
            indent=2,
            default=str,
        )
    )


# =========================================================
# 4. PRINT RESPONSE ERROR
# =========================================================

def print_response_error(
    response: httpx.Response,
) -> None:
    """
    Print the complete API error response.

    This is especially useful for HTTP 422 because it tells
    us whether validation failed in:

        Pydantic

    or:

        MLflow model-signature validation
    """

    print()

    print(
        "API ERROR RESPONSE"
    )

    print(
        "=" * 60
    )

    print(
        f"HTTP Status: "
        f"{response.status_code}"
    )

    print()


    try:

        print(
            json.dumps(
                response.json(),
                indent=2,
                default=str,
            )
        )

    except Exception:

        print(
            response.text
        )


# =========================================================
# 5. WAIT FOR FASTAPI
# =========================================================

def wait_for_api(
    client: httpx.Client,
) -> None:
    """
    Wait until FastAPI starts responding.

    Why?

    FastAPI loads the MLflow @champion model during its
    lifespan startup.

    With Uvicorn --reload the process may temporarily be
    unavailable while the model is being downloaded and
    deserialized.

    Instead of immediately failing, poll /health until:

        - API responds successfully
        OR
        - startup timeout expires
    """

    health_url = (
        f"{API_URL}/health"
    )


    print()

    print(
        "WAITING FOR PAYFLOW API"
    )

    print(
        "=" * 60
    )

    print(
        f"URL     : {health_url}"
    )

    print(
        f"Timeout : "
        f"{API_STARTUP_TIMEOUT_SECONDS:.0f} seconds"
    )


    started_at = (
        time.monotonic()
    )


    attempt = 0


    while True:

        attempt += 1


        elapsed = (
            time.monotonic()
            - started_at
        )


        if (
            elapsed
            >= API_STARTUP_TIMEOUT_SECONDS
        ):

            raise RuntimeError(
                "\nPayFlow API did not become available "
                f"within "
                f"{API_STARTUP_TIMEOUT_SECONDS:.0f} "
                "seconds.\n\n"
                "Check the Uvicorn terminal and verify "
                "that application startup completed."
            )


        try:

            response = (
                client.get(
                    health_url
                )
            )


            if response.is_success:

                print(
                    f"Attempt {attempt}: "
                    "API is available."
                )

                return


            print(
                f"Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )


        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
        ) as error:

            print(
                f"Attempt {attempt}: "
                f"waiting... "
                f"({type(error).__name__})"
            )


        time.sleep(
            API_STARTUP_POLL_SECONDS
        )


# =========================================================
# 6. REQUIRE SUCCESSFUL HTTP RESPONSE
# =========================================================

def require_success(
    response: httpx.Response,
    operation: str,
) -> None:
    """
    Convert an unsuccessful HTTP response into a readable
    PayFlow smoke-test error.

    We print the server response BEFORE raising.
    """

    if response.is_success:
        return


    print_response_error(
        response
    )


    raise RuntimeError(
        f"{operation} failed with "
        f"HTTP {response.status_code}."
    )


# =========================================================
# 7. MAIN SMOKE TEST
# =========================================================

def main() -> None:
    """
    Run the complete local Part 6 FastAPI smoke test.

    Flow:

        API availability
            ↓
        /health
            ↓
        /ready
            ↓
        /v1/model
            ↓
        /v1/predict
    """

    # =====================================================
    # LOAD ONE REAL TRANSACTION
    #
    # Do not invent categorical values manually.
    #
    # Using a real dataset row gives us valid categories
    # from the same source used during training.
    # =====================================================

    df = load_transactions(
        DATA_PATH
    )


    sample = (
        df[
            FEATURE_COLUMNS
        ]
        .head(1)
        .copy()
    )


    # =====================================================
    # SHOW TRAINING/INFERENCE DTYPES
    #
    # Very useful while debugging HTTP 422.
    #
    # It allows us to compare:
    #
    # DataFrame dtype
    #       ↓
    # Pydantic API type
    #       ↓
    # MLflow model signature
    # =====================================================

    feature_dtypes = {

        column:
            str(
                sample[
                    column
                ].dtype
            )

        for column in (
            FEATURE_COLUMNS
        )
    }


    print_json(
        "SOURCE FEATURE DTYPES",
        feature_dtypes,
    )


    # =====================================================
    # CONVERT DATAFRAME ROW TO JSON
    #
    # DataFrame.to_json() converts NumPy scalar values into
    # normal JSON-compatible Python values.
    # =====================================================

    records = json.loads(
        sample.to_json(
            orient="records"
        )
    )


    transaction = (
        records[0]
    )


    # =====================================================
    # PRINT EXACT REQUEST BEFORE SENDING IT
    #
    # If FastAPI later says HTTP 422, we can compare the
    # exact rejected payload with the error response.
    # =====================================================

    print_json(
        "REQUEST PAYLOAD",
        transaction,
    )


    # =====================================================
    # HTTP CLIENT
    # =====================================================

    with httpx.Client(
        timeout=HTTP_TIMEOUT
    ) as client:


        # =================================================
        # WAIT FOR FASTAPI
        # =================================================

        wait_for_api(
            client
        )


        # =================================================
        # HEALTH
        # =================================================

        health = (
            client.get(
                f"{API_URL}/health"
            )
        )


        require_success(
            health,
            "Health check",
        )


        print_json(
            "HEALTH",
            health.json(),
        )


        # =================================================
        # READINESS
        #
        # Health means:
        #
        #     process alive
        #
        # Readiness means:
        #
        #     production model loaded
        # =================================================

        ready = (
            client.get(
                f"{API_URL}/ready"
            )
        )


        require_success(
            ready,
            "Readiness check",
        )


        print_json(
            "READINESS",
            ready.json(),
        )


        # =================================================
        # MODEL METADATA
        #
        # Confirms:
        #
        #     @champion
        #     Version 1
        #     quality gate
        #     threshold
        # =================================================

        model = (
            client.get(
                f"{API_URL}/v1/model"
            )
        )


        require_success(
            model,
            "Model metadata request",
        )


        print_json(
            "MODEL",
            model.json(),
        )


        # =================================================
        # PREDICTION
        # =================================================

        print()

        print(
            "CALLING /v1/predict"
        )

        print(
            "=" * 60
        )


        prediction = (
            client.post(

                f"{API_URL}/v1/predict",

                json=(
                    transaction
                ),

                headers={
                    "X-Request-ID":
                        "payflow-local-test-001"
                },
            )
        )


        # =================================================
        # ALWAYS PRINT STATUS
        #
        # This makes HTTP 422 immediately obvious.
        # =================================================

        print()

        print(
            "PREDICTION HTTP STATUS"
        )

        print(
            "=" * 60
        )

        print(
            prediction.status_code
        )


        # =================================================
        # HANDLE VALIDATION ERROR
        #
        # DO NOT call raise_for_status() before printing the
        # response body.
        #
        # Otherwise httpx hides the useful FastAPI / MLflow
        # validation information behind a traceback.
        # =================================================

        if not prediction.is_success:

            print_response_error(
                prediction
            )


            print()

            print(
                "========================================"
            )

            print(
                "PREDICTION REQUEST FAILED"
            )

            print(
                "========================================"
            )

            print()

            print(
                "The API is running correctly, but the "
                "prediction request did not satisfy the "
                "inference contract."
            )

            print()

            print(
                "Inspect API ERROR RESPONSE above."
            )

            print()


            # -------------------------------------------------
            # Exit cleanly with an informative error instead of
            # producing a long httpx traceback.
            # -------------------------------------------------

            return


        # =================================================
        # SUCCESSFUL PREDICTION
        # =================================================

        print_json(
            "PREDICTION",
            prediction.json(),
        )


        # =================================================
        # FINAL RESULT
        # =================================================

        print()

        print(
            "========================================"
        )

        print(
            "PAYFLOW FASTAPI SMOKE TEST PASSED"
        )

        print(
            "========================================"
        )

        print()

        print(
            "✓ API process is alive"
        )

        print(
            "✓ Production model is ready"
        )

        print(
            "✓ MLflow @champion resolved"
        )

        print(
            "✓ Registry threshold loaded"
        )

        print(
            "✓ Prediction endpoint succeeded"
        )


# =========================================================
# 8. SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()