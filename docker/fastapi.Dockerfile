FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip bash ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH" PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade pip setuptools wheel && pip install "uv>=0.4,<0.5"

USER root
WORKDIR /app

RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

ARG BACKEND=cuda

COPY pyproject.toml ./
RUN uv pip install .[${BACKEND}]

USER root

ENV PYTHONPATH=/app

EXPOSE 8000

COPY src /app/src

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# FastAPI Dockerfile with hot reloading for development
# FROM python:3.10.11-slim AS builder

# RUN apt-get update && apt-get install -y bash curl gcc g++ git && rm -rf /var/lib/apt/lists/*

# RUN pip install uv

# WORKDIR /app

# COPY pyproject.toml ./
# RUN uv venv .venv

# ENV PATH="/app/.venv/bin:$PATH"

# RUN uv pip install .

# FROM python:3.10.11-slim AS runtime


# # Install additional FastAPI dependencies
# RUN uv pip install --system fastapi uvicorn[standard] python-multipart

# # Create non-root user for security
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# # Expose the port FastAPI will run on
# EXPOSE 8000

# Command to run FastAPI with hot reloading
# --reload enables automatic reloading when code changes
# --host 0.0.0.0 allows external connections
# --port 8000 sets the port
# CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
