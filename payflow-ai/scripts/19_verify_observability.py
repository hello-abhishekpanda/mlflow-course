from __future__ import annotations

import json
import os
import time

import httpx


# =========================================================
# PAYFLOW AI — PART 8
#
# OBSERVABILITY STACK VERIFICATION
# =========================================================


LOKI_URL = os.getenv(
    "PAYFLOW_LOKI_URL",
    "http://127.0.0.1:3100",
)

GRAFANA_URL = os.getenv(
    "PAYFLOW_GRAFANA_URL",
    "http://127.0.0.1:3000",
)

FASTAPI_URL = os.getenv(
    "PAYFLOW_API_URL",
    "http://127.0.0.1:8001",
)

MLFLOW_URL = os.getenv(
    "PAYFLOW_DOCKER_MLFLOW_URI",
    "http://127.0.0.1:5050",
)


def check(
    name: str,
    url: str,
    expected_status: int = 200,
) -> None:

    response = httpx.get(
        url,
        timeout=20.0,
    )


    if (
        response.status_code
        != expected_status
    ):

        raise RuntimeError(
            f"{name} failed: "
            f"HTTP {response.status_code}"
        )


    print(
        f"✅ {name}"
    )


print()
print("=" * 72)
print("PAYFLOW OBSERVABILITY VERIFICATION")
print("=" * 72)


check(
    "MLflow",
    f"{MLFLOW_URL}/health",
)

check(
    "Loki",
    f"{LOKI_URL}/ready",
)

check(
    "Grafana",
    f"{GRAFANA_URL}/api/health",
)

check(
    "FastAPI health",
    f"{FASTAPI_URL}/health",
)

check(
    "FastAPI readiness",
    f"{FASTAPI_URL}/ready",
)


# =========================================================
# GENERATE REQUEST LOGS + TRACES
# =========================================================

print()
print(
    "Generating observable traffic..."
)


for endpoint in [
    "/health",
    "/ready",
    "/v1/model",
]:

    response = httpx.get(
        FASTAPI_URL + endpoint,
        timeout=20.0,
    )

    print(
        f"{endpoint}: "
        f"{response.status_code}"
    )


time.sleep(
    3
)


print()
print("=" * 72)
print("OBSERVABILITY INFRASTRUCTURE PASSED")
print("=" * 72)

print(
    f"Grafana : {GRAFANA_URL}"
)

print(
    f"MLflow  : {MLFLOW_URL}"
)

print()
print(
    "Next manual checks:"
)

print(
    "1. Grafana -> Explore -> Loki"
)

print(
    "2. MLflow -> payflow-production-traces -> Traces"
)