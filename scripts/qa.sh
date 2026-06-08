#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Formatting with ruff..."
uv run ruff format .

echo "==> Linting with ruff..."
uv run ruff check --fix .

echo "==> Type checking with pyright..."
uv run pyright

echo "==> Running tests..."
uv run pytest

echo "==> QA passed!"
