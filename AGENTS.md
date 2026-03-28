# Repository Guidelines

## Commands

### Development Setup

```bash
uv sync
```

### Code Quality and Documentation

**Important**: Run these before committing or creating a PR.

1. **format, lint, and test**: Use `local-qa` skill.
2. **Documentation build** (if any public API changes): `uv run mkdocs build`

## Architecture

### Key Dependencies

- **pdmt5**: Pandas-based data handler for MetaTrader 5 (core library)
- **typer**: CLI framework for building command-line interfaces
- **click**: Parameter type customization for CLI options
- **pandas**: Core data manipulation and analysis

### Package Structure

- `mt5cli/`: Main package directory
  - `__init__.py`: Package initialization and exports (`detect_format`, `export_dataframe`)
  - `cli.py`: CLI application with typer-based commands for data export
  - `__main__.py`: Entry point for `python -m mt5cli`
- `tests/`: Comprehensive test suite (pytest-based)
  - `test_cli.py`: Tests for CLI commands, parameter types, and export functions
- `docs/`: MkDocs documentation with API reference
  - `docs/index.md`: Main documentation
  - `docs/api/`: Auto-generated API documentation for all modules
- Modern Python packaging with `pyproject.toml` and uv dependency management

### Quality Standards

- Type hints required (pyright strict mode)
- Comprehensive linting with 35+ rule categories (ruff)
- Test coverage tracking with 100% (pytest-cov)
- Parametrized tests for input/result matrices using `pytest.mark.parametrize` (pytest)
- Test doubles (mocks, stubs) using `pytest_mock` for external dependencies (pytest-mock)
- Pydantic models for data validation and configuration

### Documentation workflow

1. Add Google-style docstrings to functions/classes
2. Local preview: `uv run mkdocs serve`
3. Build: `uv run mkdocs build`
4. Deploy: `uv run mkdocs gh-deploy`

## Commit & Pull Request Guidelines

- Run QA checks using `local-qa` skill before committing or creating a PR.
- Branch names use appropriate prefixes on creation (e.g., `feature/...`, `bugfix/...`, `refactor/...`, `docs/...`, `chore/...`).
- When instructed to create a PR, create it as a draft with appropriate labels by default.

## Code Design Principles

Always prefer the simplest design that works.

- **KISS**: Choose straightforward solutions and avoid unnecessary abstraction.
- **DRY**: Remove duplication when it improves clarity and maintainability.
- **YAGNI**: Do not add features, hooks, or flexibility until they are needed.
- **SOLID/Clean Code**: Apply these as tools, only when they keep the design simpler and easier to change.

## Development Methodology

Keep delivery incremental, test-backed, and easy to review.

- Make small, safe, reversible changes.
- Prefer `Red -> Green -> Refactor`.
- Do not mix feature work and refactoring in the same commit.
- Refactor when it improves clarity or removes real duplication (Rule of Three).
- Keep tests fast, focused, and self-validating.
