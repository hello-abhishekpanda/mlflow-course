# =========================================================
# PAYFLOW MLFLOW SERVER
#
# MLflow Tracking Server
# +
# PostgreSQL backend
# +
# MinIO artifacts
# =========================================================

FROM ghcr.io/mlflow/mlflow:v3.16.0


# =========================================================
# PostgreSQL driver
#
# psycopg2-binary:
#     MLflow -> PostgreSQL
#
# boto3:
#     MLflow -> MinIO through S3-compatible API
# =========================================================

RUN pip install \
    --no-cache-dir \
    psycopg2-binary \
    boto3