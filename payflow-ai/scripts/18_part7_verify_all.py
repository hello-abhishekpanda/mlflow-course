from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import mlflow

from dotenv import load_dotenv
from mlflow import MlflowClient


# =========================================================
# PAYFLOW AI — PART 7
#
# COMPLETE ONE-COMMAND VERIFICATION
#
# Checks:
#
#   Docker infrastructure
#   PostgreSQL
#   MinIO
#   MLflow Tracking
#   MLflow artifact round-trip
#   Model Registry
#   @champion
#   governance tags
#   FastAPI
#   prediction
#   restart persistence
# =========================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


ENV_FILE = (
    PROJECT_ROOT
    / ".env.docker"
)


load_dotenv(
    ENV_FILE
)


SOURCE_MLFLOW = (
    "http://127.0.0.1:5002"
)

TARGET_MLFLOW = (
    "http://127.0.0.1:5050"
)

MINIO_CONSOLE = (
    "http://127.0.0.1:9001"
)

FASTAPI_URL = (
    "http://127.0.0.1:8001"
)


MODEL_NAME = (
    "payflow-payment-success"
)

MODEL_ALIAS = (
    "champion"
)


# Host clients use the MLflow proxy instead of Docker-only
# minio:9000.
os.environ[
    "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"
] = "false"

os.environ[
    "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"
] = "false"


def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run(
    command: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:

    merged_env = os.environ.copy()

    if env:
        merged_env.update(env)

    print()
    print("$ " + " ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=merged_env,
        text=True,
        capture_output=capture,
    )

    if capture:

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)


    if result.returncode != 0:

        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
        )

    return result


def compose(
    *args: str,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:

    return run(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV_FILE),
            *args,
        ],
        capture=capture,
    )


def wait_for(
    url: str,
    *,
    timeout: int = 180,
) -> httpx.Response:

    deadline = (
        time.time()
        + timeout
    )

    last_error: Exception | None = None


    while time.time() < deadline:

        try:

            response = httpx.get(
                url,
                timeout=10.0,
            )

            if response.status_code < 500:

                return response

        except Exception as exc:

            last_error = exc


        time.sleep(2)


    raise RuntimeError(
        f"Timed out waiting for {url}. "
        f"Last error: {last_error}"
    )


# =========================================================
# 1. FILE CHECK
# =========================================================

section(
    "1. VERIFY PROJECT FILES"
)


required = [

    ".env.docker",

    "docker-compose.yml",

    "docker/mlflow/Dockerfile",

    "docker/fastapi/Dockerfile",

    "docker/fastapi/requirements.txt",

    "scripts/15_test_fastapi.py",

    "scripts/16_test_docker_mlflow.py",

    "scripts/17_bootstrap_docker_registry.py",
]


for relative_path in required:

    path = (
        PROJECT_ROOT
        / relative_path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    print(
        f"✅ {relative_path}"
    )


# =========================================================
# 2. START CORE INFRASTRUCTURE
# =========================================================

section(
    "2. START POSTGRESQL + MINIO + MLFLOW"
)


compose(
    "up",
    "-d",
    "postgres",
    "minio",
    "minio-init",
    "mlflow",
)


wait_for(
    f"{TARGET_MLFLOW}/health"
)


print(
    "✅ Docker MLflow healthy"
)


# =========================================================
# 3. CONTAINER STATUS
# =========================================================

section(
    "3. DOCKER STATUS"
)


compose(
    "ps"
)


# =========================================================
# 4. STORAGE SMOKE TEST
# =========================================================

section(
    "4. POSTGRESQL + MINIO STORAGE TEST"
)


run(
    [
        sys.executable,
        "scripts/16_test_docker_mlflow.py",
    ],
    env={
        "PYTHONPATH":
            str(PROJECT_ROOT / "src"),

        "PAYFLOW_DOCKER_MLFLOW_URI":
            TARGET_MLFLOW,
    },
)


print(
    "✅ MLflow storage test passed"
)


# =========================================================
# 5. POSTGRESQL VERIFICATION
# =========================================================

section(
    "5. POSTGRESQL VERIFICATION"
)


postgres_query = """
SELECT
    experiment_id,
    name,
    artifact_location
FROM experiments
ORDER BY experiment_id;
"""


postgres = compose(
    "exec",
    "-T",
    "postgres",
    "psql",
    "-U",
    os.getenv(
        "POSTGRES_USER",
        "mlflow",
    ),
    "-d",
    os.getenv(
        "POSTGRES_DB",
        "mlflow",
    ),
    "-P",
    "pager=off",
    "-c",
    postgres_query,
    capture=True,
)


if (
    "payflow-docker-smoke"
    not in postgres.stdout
):

    raise RuntimeError(
        "Smoke experiment not found "
        "inside PostgreSQL."
    )


print(
    "✅ PostgreSQL metadata verified"
)


# =========================================================
# 6. MINIO VERIFICATION
# =========================================================

section(
    "6. MINIO VERIFICATION"
)


minio_command = (
    '/usr/bin/mc alias set '
    'payflow '
    'http://minio:9000 '
    '"$MINIO_ROOT_USER" '
    '"$MINIO_ROOT_PASSWORD" '
    '>/dev/null '
    '&& '
    '/usr/bin/mc ls '
    '--recursive '
    'payflow/"$MLFLOW_BUCKET" '
    '| tail -n 50'
)


minio = compose(
    "run",
    "--rm",
    "--no-deps",
    "--entrypoint",
    "/bin/sh",
    "minio-init",
    "-c",
    minio_command,
    capture=True,
)


if not minio.stdout.strip():

    raise RuntimeError(
        "MinIO bucket appears empty."
    )


print(
    "✅ MinIO artifacts verified"
)


# =========================================================
# 7. LEGACY SOURCE REGISTRY
# =========================================================

section(
    "7. VERIFY LEGACY MLFLOW :5002"
)


try:

    wait_for(
        f"{SOURCE_MLFLOW}/health",
        timeout=10,
    )

except Exception as exc:

    raise RuntimeError(
        "\nLegacy MLflow is not running.\n\n"
        "Start it in a second terminal:\n\n"
        "mlflow server \\\n"
        "  --host 127.0.0.1 \\\n"
        "  --port 5002 \\\n"
        "  --backend-store-uri sqlite:///mlflow.db \\\n"
        "  --artifacts-destination ./mlartifacts \\\n"
        "  --allowed-hosts "
        "\"localhost:*,127.0.0.1:*\"\n"
    ) from exc


print(
    "✅ Legacy MLflow :5002 healthy"
)


# =========================================================
# 8. BOOTSTRAP CHAMPION
# =========================================================

section(
    "8. BOOTSTRAP @CHAMPION"
)


run(
    [
        sys.executable,
        "scripts/17_bootstrap_docker_registry.py",
    ],
    env={
        "PYTHONPATH":
            str(PROJECT_ROOT / "src"),

        "PAYFLOW_SOURCE_MLFLOW_URI":
            SOURCE_MLFLOW,

        "PAYFLOW_DOCKER_MLFLOW_URI":
            TARGET_MLFLOW,
    },
)


# =========================================================
# 9. VERIFY REGISTRY
# =========================================================

section(
    "9. VERIFY DOCKER REGISTRY"
)


client = MlflowClient(
    tracking_uri=TARGET_MLFLOW,
    registry_uri=TARGET_MLFLOW,
)


version = (
    client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )
)


tags = dict(
    version.tags or {}
)


print(
    f"Model     : {version.name}"
)

print(
    f"Version   : {version.version}"
)

print(
    f"Alias     : @{MODEL_ALIAS}"
)

print(
    f"Threshold : "
    f"{tags.get('failure_threshold')}"
)

print(
    f"Gate      : "
    f"{tags.get('quality_gate')}"
)

print(
    f"Eligible  : "
    f"{tags.get('registry_eligible')}"
)


if (
    tags.get(
        "quality_gate",
        ""
    ).upper()
    != "PASSED"
):

    raise RuntimeError(
        "quality_gate is not PASSED."
    )


if (
    tags.get(
        "registry_eligible",
        ""
    ).lower()
    != "true"
):

    raise RuntimeError(
        "registry_eligible is not true."
    )


if not tags.get(
    "failure_threshold"
):

    raise RuntimeError(
        "failure_threshold missing."
    )


print(
    "✅ Registry governance verified"
)


# =========================================================
# 10. START FASTAPI
# =========================================================

section(
    "10. START FASTAPI"
)


compose(
    "up",
    "-d",
    "fastapi",
)


wait_for(
    f"{FASTAPI_URL}/ready",
    timeout=180,
)


print(
    "✅ FastAPI ready"
)


# =========================================================
# 11. API OPERATIONAL CHECKS
# =========================================================

section(
    "11. FASTAPI OPERATIONAL ENDPOINTS"
)


for endpoint in [
    "/health",
    "/ready",
    "/v1/model",
]:

    response = httpx.get(
        FASTAPI_URL + endpoint,
        timeout=30.0,
    )

    print()
    print(
        f"GET {endpoint}"
    )

    print(
        f"HTTP {response.status_code}"
    )

    try:

        print(
            json.dumps(
                response.json(),
                indent=2,
            )
        )

    except Exception:

        print(
            response.text
        )


    if response.status_code != 200:

        raise RuntimeError(
            f"{endpoint} failed."
        )


print(
    "✅ Operational endpoints verified"
)


# =========================================================
# 12. REAL PREDICTION SMOKE TEST
# =========================================================

section(
    "12. REAL PREDICTION TEST"
)


run(
    [
        sys.executable,
        "scripts/15_test_fastapi.py",
    ],
    env={
        "PYTHONPATH":
            str(PROJECT_ROOT / "src"),

        "PAYFLOW_API_URL":
            FASTAPI_URL,
    },
)


print(
    "✅ Prediction smoke test passed"
)


# =========================================================
# 13. PERSISTENCE TEST
# =========================================================

section(
    "13. RESTART + PERSISTENCE TEST"
)


compose(
    "restart"
)


wait_for(
    f"{TARGET_MLFLOW}/health",
    timeout=180,
)


wait_for(
    f"{FASTAPI_URL}/ready",
    timeout=240,
)


after_restart_client = (
    MlflowClient(
        tracking_uri=TARGET_MLFLOW,
        registry_uri=TARGET_MLFLOW,
    )
)


after_restart_version = (
    after_restart_client
    .get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )
)


print(
    f"Champion survived restart: "
    f"Version "
    f"{after_restart_version.version}"
)


print(
    "✅ Registry persistence verified"
)


# =========================================================
# 14. FINAL STATUS
# =========================================================

section(
    "14. FINAL DOCKER STATUS"
)


compose(
    "ps"
)


# =========================================================
# SUCCESS
# =========================================================

section(
    "PART 7 END-TO-END VERIFICATION PASSED"
)


print(
    """
✅ PostgreSQL backend store
✅ MinIO artifact store
✅ Docker MLflow
✅ Artifact upload
✅ Artifact download
✅ PostgreSQL metadata
✅ MinIO objects
✅ Registry Version 1
✅ @champion
✅ Governance tags
✅ Failure threshold
✅ Docker FastAPI
✅ API readiness
✅ Model metadata
✅ Real prediction
✅ Docker restart
✅ Persistent Registry
"""
)


print(
    "UI CHECKS"
)

print(
    f"Legacy MLflow : {SOURCE_MLFLOW}"
)

print(
    f"Docker MLflow : {TARGET_MLFLOW}"
)

print(
    f"MinIO Console : {MINIO_CONSOLE}"
)

print(
    f"FastAPI Ready : {FASTAPI_URL}/ready"
)

print(
    f"FastAPI Model : {FASTAPI_URL}/v1/model"
)

print(
    f"FastAPI Docs  : {FASTAPI_URL}/docs"
)