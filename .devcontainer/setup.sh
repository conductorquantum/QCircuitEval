#!/usr/bin/env bash
set -euo pipefail

git config --global --add safe.directory "${PWD}"
uv sync --extra dev --extra docs
uv run python -m pre_commit install --install-hooks
