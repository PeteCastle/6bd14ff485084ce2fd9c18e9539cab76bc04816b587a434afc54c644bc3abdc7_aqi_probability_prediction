FROM apache/airflow:slim-3.0.3-python3.10

USER root

ARG BACKEND=cpu  # default backend
RUN apt-get update && apt-get install -y bash

USER airflow

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv .venv

ENV PATH="/app/.venv/bin:$PATH"

ARG BACKEND=cpu  # default backend

RUN uv pip install .[${BACKEND}]

# COPY src /opt/airflow/src
COPY data /opt/airflow/data
USER root
RUN chmod -R a+w /opt/airflow/data
USER airflow
# COPY airflow/dags /opt/airflow/dags
COPY deploy/airflow/plugins /opt/airflow/plugins
