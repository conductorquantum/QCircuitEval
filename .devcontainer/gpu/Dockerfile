# syntax=docker/dockerfile:1.7

ARG CUDAQ_IMAGE=nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.14.2
FROM ${CUDAQ_IMAGE}

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git sudo \
    && rm -rf /var/lib/apt/lists/* \
    && if ! id -u vscode >/dev/null 2>&1; then useradd --create-home --shell /bin/bash vscode; fi \
    && usermod -aG sudo vscode \
    && printf "vscode ALL=(ALL) NOPASSWD:ALL\n" >/etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=.venv

USER vscode
