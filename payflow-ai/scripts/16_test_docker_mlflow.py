from __future__ import annotations

import os
from pathlib import Path


# =========================================================
# PAYFLOW AI — PART 7
#
# MLflow Docker Storage Smoke Test
#
# Proves:
#
#   MLflow metadata -> PostgreSQL
#   MLflow artifacts -> MinIO
#   MinIO artifact -> MLflow -> local client
# =========================================================


# =========================================================
# IMPORTANT FOR LOCAL DOCKER + MINIO
#
# MinIO is known to containers as:
#
#     http://minio:9000
#
# The host-side Python process cannot use that Docker-only
# hostname directly.
#
# Force artifact traffic through the MLflow tracking server.
# =========================================================

os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"


import mlflow

from dotenv import load_dotenv
from mlflow.artifacts import download_artifacts


load_dotenv()


TRACKING_URI = os.getenv(
    "PAYFLOW_DOCKER_MLFLOW_URI",
    "http://127.0.0.1:5050",
)

EXPERIMENT_NAME = "payflow-docker-smoke"


# =========================================================
# CONNECT TO DOCKER MLFLOW
# =========================================================

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


print()
print("=" * 72)
print("PAYFLOW DOCKER MLFLOW STORAGE TEST")
print("=" * 72)

print(f"Tracking URI : {TRACKING_URI}")


# =========================================================
# CREATE SMOKE RUN
# =========================================================

with mlflow.start_run(
    run_name="docker-storage-smoke-test"
) as run:

    # Stored through the PostgreSQL backend.
    mlflow.log_param(
        "storage_backend",
        "postgresql",
    )

    mlflow.log_param(
        "artifact_backend",
        "minio",
    )

    mlflow.log_metric(
        "smoke_metric",
        1.0,
    )

    mlflow.set_tags(
        {
            "project": "payflow-ai",
            "stage": "docker-infrastructure-test",
            "backend_store": "postgresql",
            "artifact_store": "minio",
        }
    )

    # Stored in MinIO through MLflow's artifact proxy.
    mlflow.log_text(
        (
            "PayFlow Docker MLflow artifact test.\n"
            "Backend Store: PostgreSQL\n"
            "Artifact Store: MinIO\n"
        ),
        "smoke/hello.txt",
    )

    run_id = run.info.run_id


print()
print("RUN CREATED")
print("=" * 72)
print(f"Experiment : {EXPERIMENT_NAME}")
print(f"Run ID     : {run_id}")


# =========================================================
# DOWNLOAD ARTIFACT BACK THROUGH MLFLOW
# =========================================================

print()
print("Downloading artifact through MLflow...")


downloaded_file = download_artifacts(
    run_id=run_id,
    artifact_path="smoke/hello.txt",
)


downloaded_path = Path(downloaded_file)


if not downloaded_path.exists():
    raise RuntimeError(
        "Artifact download returned a path, but the "
        "downloaded file does not exist."
    )


contents = downloaded_path.read_text(
    encoding="utf-8"
)


print()
print("ARTIFACT DOWNLOADED")
print("=" * 72)
print(f"Path: {downloaded_path}")

print()
print(contents)


if "PayFlow Docker MLflow artifact test." not in contents:
    raise RuntimeError(
        "Downloaded artifact contents were incorrect."
    )


print()
print("=" * 72)
print("DOCKER MLFLOW STORAGE TEST PASSED")
print("=" * 72)

print("PostgreSQL metadata write : PASS")
print("MinIO artifact upload     : PASS")
print("MinIO artifact download   : PASS")
print("MLflow tracking server    : PASS")