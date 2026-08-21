# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    HOME=/tmp/qceval-home \
    XDG_CACHE_HOME=/tmp/qceval-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1

WORKDIR /opt/qceval

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --locked --no-dev \
    && groupadd --system qceval \
    && useradd --system --gid qceval --home-dir /home/qceval --create-home qceval \
    && mkdir -p /results /tmp/qceval-home /tmp/qceval-cache /tmp/matplotlib /tmp/cuda-cache /tmp/uv-cache /tmp/pycache \
    && chown -R qceval:qceval /results /home/qceval /tmp/qceval-home /tmp/qceval-cache /tmp/matplotlib /tmp/cuda-cache /tmp/uv-cache /tmp/pycache

USER qceval
VOLUME ["/results"]

ENTRYPOINT ["uv", "run", "--no-sync"]
CMD ["qceval", "--help"]
