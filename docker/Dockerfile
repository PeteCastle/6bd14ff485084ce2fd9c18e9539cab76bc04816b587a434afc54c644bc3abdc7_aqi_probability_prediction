FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y bash
ARG BACKEND=cpu  # default backend

ENV PYTORCH_ENABLE_MPS_FALLBACK=0
 # Disable MPS fallback for PyTorch

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv .venv

ENV PATH="/app/.venv/bin:$PATH"

RUN uv pip install .[${BACKEND}]

COPY src ./src

CMD ["python", "-m", "src.run_pipeline"]

