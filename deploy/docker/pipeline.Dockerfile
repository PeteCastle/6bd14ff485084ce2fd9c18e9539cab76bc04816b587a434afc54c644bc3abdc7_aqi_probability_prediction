FROM python:3.10.11-slim AS builder

RUN apt-get update && apt-get install -y bash curl gcc g++ git && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv venv .venv

ENV PATH="/app/.venv/bin:$PATH"

ARG BACKEND=cpu
RUN uv pip install .[${BACKEND}]

FROM python:3.10.11-slim AS runtime

ENV PYTORCH_ENABLE_MPS_FALLBACK=0
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src

ENTRYPOINT ["python", "-m", "src.run_pipeline"]
CMD []
