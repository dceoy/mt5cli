---
name: local-qa
description: Run local QA including formatting, linting, and testing for the repository. Use whenever any file has been updated.
---

# Local QA (format, lint, and test)

Run `scripts/qa.sh` by its path relative to this skill's directory; the script changes to the repository root itself.

## Procedure

1. Run `scripts/qa.sh`. It formats and fixes files in place, so check `git status` afterward.
2. Summarize the result: pass or fail, major warnings, and the files it modified.
3. If a tool is missing, install it with the project's package manager first (`uv`, `pnpm`), then a system package manager, then a language-specific installer, and rerun. If installation fails or needs privileges you lack, report what you tried and the exact error, then stop.
4. If QA fails for another reason, fix the root cause and rerun until it passes or you hit a blocker. Do not bypass or weaken checks.
5. Run only commands needed for QA, fixes, and tool installation.
