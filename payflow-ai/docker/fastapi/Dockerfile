# =========================================================
# PAYFLOW FASTAPI INFERENCE SERVICE
#
# Part 7:
# Dockerized FastAPI service that loads the approved
# MLflow @champion model.
# =========================================================


# =========================================================
# 1. BASE PYTHON IMAGE
#
# Use a small Python image for the inference container.
# =========================================================

FROM python:3.12-slim


# =========================================================
# 2. PYTHON RUNTIME SETTINGS
#
# PYTHONDONTWRITEBYTECODE:
#     Do not create .pyc files.
#
# PYTHONUNBUFFERED:
#     Send logs directly to Docker logs without buffering.
# =========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# =========================================================
# 3. APPLICATION WORKING DIRECTORY
# =========================================================

WORKDIR /app


# =========================================================
# 4. COPY DEPENDENCIES FIRST
#
# Local:
#
#     docker/fastapi/requirements.txt
#
# Container:
#
#     /tmp/requirements.txt
#
# We copy requirements separately so Docker can cache the
# dependency installation layer.
# =========================================================

COPY docker/fastapi/requirements.txt /tmp/requirements.txt


# =========================================================
# 5. INSTALL PYTHON DEPENDENCIES
#
# --no-cache-dir keeps the final image smaller.
# =========================================================

RUN pip install \
    --no-cache-dir \
    --upgrade pip \
    && pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt


# =========================================================
# 6. COPY PAYFLOW SOURCE CODE
#
# Local:
#
#     src/
#
# Container:
#
#     /app/src/
# =========================================================

COPY src /app/src


# =========================================================
# 7. PYTHON SRC-LAYOUT IMPORT PATH
#
# Our package lives at:
#
#     /app/src/payflow
#
# Adding /app/src to PYTHONPATH allows:
#
#     import payflow
#
# to work inside Docker.
# =========================================================

ENV PYTHONPATH=/app/src


# =========================================================
# 8. DOCUMENT FASTAPI PORT
#
# FastAPI listens internally on port 8000.
#
# Docker Compose exposes:
#
#     localhost:8001
#            ↓
#     container:8000
# =========================================================

EXPOSE 8000


# =========================================================
# 9. START FASTAPI
#
# IMPORTANT:
#
# Dockerfile instructions must be valid Docker syntax.
#
# Use JSON-array CMD on ONE LINE.
#
# Do NOT use --reload inside the container.
# =========================================================

CMD ["uvicorn", "payflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]