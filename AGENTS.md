# Repository Guidelines

## Project Structure & Module Organization

`mt5cli/` contains the package source. Important modules include `cli.py` for the Typer command-line app, `client.py` and `sdk.py` for public MT5 client/session APIs, `history.py` for SQLite history collection, `storage.py` and `converters.py` for export behavior, and `schemas.py` for normalized dataset contracts. `tests/` holds pytest coverage for CLI behavior, SDK contracts, trading helpers, history, and utilities. `docs/` and `mkdocs.yml` define the MkDocs site and API reference. `skills/mt5cli/SKILL.md` documents the mt5cli agent skill.

## Build, Test, and Development Commands

- `uv sync` installs runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run mt5cli --help` runs the local CLI entry point.
- `uv run ruff format .` formats Python files.
- `uv run ruff check --fix .` lints and applies safe fixes.
- `uv run pyright .` runs strict type checking.
- `uv run pytest` runs doctests, branch coverage, and the test suite.
- `uv run mkdocs serve` previews documentation locally; `uv run mkdocs build` validates the docs build.

Use `.agents/skills/local-qa/SKILL.md` for pre-handoff QA. It runs `.agents/skills/local-qa/scripts/qa.sh`, which formats, lints, type-checks, tests, formats Markdown, and checks GitHub workflows.

## Coding Style & Naming Conventions

Target Python `>=3.11,<3.14`. Use Ruff’s configured 88-character line length and Google-style docstrings. Pyright is strict, so prefer explicit public type annotations and narrow exception handling. Keep module, function, and variable names in `snake_case`; classes and enums use `PascalCase`. Preserve the package’s small, typed helper style rather than adding broad abstractions.

## Design Principles

Apply KISS, DRY, and YAGNI when changing code. Prefer the simplest implementation that satisfies the current CLI/API contract. Remove duplication when shared behavior is already proven by at least two concrete call sites, but avoid generic helpers for speculative reuse. Do not add configuration flags, extension hooks, or alternate backends until a real repository use case requires them.

## Testing Guidelines

Tests use pytest, pytest-mock, doctests, and pytest-cov. Test files should match `tests/test_*.py`, classes `Test*`, and functions `test_*`. Coverage is configured with `fail_under = 100`, so add focused tests for every behavior change. Mock MT5/pdmt5 boundaries; do not require a live MetaTrader terminal in unit tests.

## Commit & Pull Request Guidelines

Recent history uses concise imperative commits, sometimes with conventional prefixes such as `feat:` or `chore:` and PR numbers appended by GitHub. Keep commits scoped to one logical change. Pull requests should describe behavior changes, note tests run, link related issues, and call out MT5/live-trading risk where relevant.

## Security & Configuration Tips

Never commit account credentials, broker passwords, exported private data, or local `.venv` contents. Treat `order_send` and CLI `order-send --yes` as live execution paths; gate examples and tests so they cannot place real trades accidentally.
