# Repository Guidelines

## Build, Test, and Development Commands

Use `.agents/skills/local-qa/SKILL.md` for pre-handoff QA. It runs `.agents/skills/local-qa/scripts/qa.sh`, which formats, lints, type-checks, tests, formats Markdown, and checks GitHub workflows.

`bats --verbose-run ./tests/cli/` runs the Windows/MT5-oriented CLI tests against the installed `mt5cli` entry point. These require a Windows environment where `uv sync` can install MetaTrader5; state clearly when they were not run.

`skills/mt5cli/SKILL.md` documents the mt5cli agent skill.

## Coding Style & Naming Conventions

Preserve the package’s small, typed helper style rather than adding broad abstractions.

## Design Principles

Apply KISS, DRY, and YAGNI when changing code. Prefer the simplest implementation that satisfies the current CLI/API contract. Remove duplication when shared behavior is already proven by at least two concrete call sites, but avoid generic helpers for speculative reuse. Do not add configuration flags, extension hooks, or alternate backends until a real repository use case requires them.

## Testing Guidelines

Parametrize unit tests with `pytest.mark.parametrize` when the same behavior needs coverage across multiple inputs. Mock MT5/pdmt5 boundaries; do not require a live MetaTrader terminal in unit tests.

## Commit & Pull Request Guidelines

Recent history uses concise imperative commits, sometimes with conventional prefixes such as `feat:` or `chore:` and PR numbers appended by GitHub. Keep commits scoped to one logical change. Pull requests should describe behavior changes, note tests run, link related issues, and call out MT5/live-trading risk where relevant.

## Security & Configuration Tips

Never commit account credentials, broker passwords, exported private data, or local `.venv` contents. Treat `order_send` and CLI `order-send --yes` as live execution paths; gate examples and tests so they cannot place real trades accidentally.
