# local-qa

Run local QA checks (format, lint, test) on the repository.

## When to use

After making changes to repository files, run `scripts/qa.sh` to validate formatting, linting, and tests.

## Steps

1. Execute `scripts/qa.sh` and capture the results.
2. Report successes, failures, warnings, and any modified files.

## If tools are missing

Install them following this priority order:

1. Project package managers (`uv`, `poetry`, npm scripts)
2. System package managers (`brew`, `apt`)
3. Language-specific installers (`pipx`, `pip`, `npm`, `go install`)

## Constraints

- Only execute QA and tool-installation commands.
- If installation fails or requires unavailable privileges, report the attempt and exact failure, then stop.
