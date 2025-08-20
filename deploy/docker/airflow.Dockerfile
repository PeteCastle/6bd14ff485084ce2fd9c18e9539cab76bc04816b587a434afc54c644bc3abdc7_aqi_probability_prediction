FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip bash ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH" PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade pip setuptools wheel && pip install "uv>=0.4,<0.5"

USER root
WORKDIR /opt/airflow

# project venv
RUN uv venv /opt/airflow/.venv
ENV PATH="/opt/airflow/.venv/bin:$PATH"
ENV AIRFLOW_HOME=/opt/airflow

ARG BACKEND=cuda
COPY pyproject.toml ./
RUN uv pip install .[${BACKEND}]

USER root

COPY data /opt/airflow/data
# USER root

# USER airflow
# COPY airflow/dags /opt/airflow/dags
COPY deploy/airflow/plugins /opt/airflow/plugins

ARG AIRFLOW_UID=50000
RUN useradd -u ${AIRFLOW_UID} -g 0 -m -s /bin/bash airflow

# ensure airflow can write to /opt/airflow/*
RUN mkdir -p /opt/airflow/{dags,logs,plugins,data} \
 && chown -R ${AIRFLOW_UID}:0 /opt/airflow \
 && chmod -R g+rwXs /opt/airflow
# optional: new files default to group-writable
ENV UMASK=002

# Ensure project root is importable
ENV PYTHONPATH=/opt/airflow


USER airflow