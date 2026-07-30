#!/usr/bin/env bats

# CLI integration tests for the installed mt5cli entry point.
#
# These cases only exercise paths that never connect to MetaTrader 5: help
# output, SQLite-only commands, and validation failures. No case passes --yes,
# so the suite can never place a live trade.

setup_file() {
  set -euo pipefail
  uv sync
}

teardown_file() {
  :
}

seed_rate_database() {
  local database="$1"
  cat > "${BATS_TEST_TMPDIR}/seed_rates.py" << 'PY'
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as conn:
    conn.execute('CREATE TABLE "rate_EURUSD__M1_1"(time TEXT, close REAL)')
    conn.executemany(
        'INSERT INTO "rate_EURUSD__M1_1"(time, close) VALUES (?, ?)',
        [("2024-01-01T00:00:00+00:00", 1.0), ("2024-01-01T00:02:00+00:00", 1.1)],
    )
PY
  uv run python "${BATS_TEST_TMPDIR}/seed_rates.py" "${database}"
}

@test "pass with \"mt5cli --help\"" {
  run uv run mt5cli --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"mt5cli collect-history --help\"" {
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/history.db" collect-history --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"mt5cli order-send --help\"" {
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/out.csv" order-send --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"mt5cli grafana-schema\"" {
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/grafana.db" grafana-schema
  [[ "${status}" -eq 0 ]]
  [[ -f "${BATS_TEST_TMPDIR}/grafana.db" ]]
}

@test "pass with \"mt5cli grafana-schema\" run twice" {
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/grafana.db" grafana-schema
  [[ "${status}" -eq 0 ]]
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/grafana.db" grafana-schema
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"mt5cli grafana-schema --publish-copy\"" {
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/grafana.db" \
    grafana-schema \
    --publish-copy "${BATS_TEST_TMPDIR}/published.db"
  [[ "${status}" -eq 0 ]]
  [[ -f "${BATS_TEST_TMPDIR}/published.db" ]]
}

@test "pass with \"mt5cli history-gaps\"" {
  seed_rate_database "${BATS_TEST_TMPDIR}/history.db"
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/gaps.json" \
    history-gaps \
    --sqlite3 "${BATS_TEST_TMPDIR}/history.db"
  [[ "${status}" -eq 0 ]]
  [[ -f "${BATS_TEST_TMPDIR}/gaps.json" ]]
}

@test "fail with \"mt5cli history-gaps\" for a missing database" {
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/gaps.json" \
    history-gaps \
    --sqlite3 "${BATS_TEST_TMPDIR}/no-such.db"
  [[ "${status}" -ne 0 ]]
  [[ ! -e "${BATS_TEST_TMPDIR}/no-such.db" ]]
}

@test "fail with \"mt5cli order-send\" without --yes" {
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/out.csv" \
    order-send \
    --request '{"action": 1, "symbol": "EURUSD"}'
  [[ "${status}" -ne 0 ]]
  [[ ! -e "${BATS_TEST_TMPDIR}/out.csv" ]]
}

@test "fail with \"mt5cli close-positions\" without a symbol or ticket" {
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/close.json" \
    close-positions
  [[ "${status}" -ne 0 ]]
  [[ ! -e "${BATS_TEST_TMPDIR}/close.json" ]]
}

@test "fail with \"mt5cli collect-history\" without SQLite3 output" {
  run uv run mt5cli \
    --output "${BATS_TEST_TMPDIR}/history.csv" \
    collect-history \
    --symbol EURUSD \
    --date-from 2024-01-01 \
    --date-to 2024-02-01
  [[ "${status}" -ne 0 ]]
  [[ ! -e "${BATS_TEST_TMPDIR}/history.csv" ]]
}

@test "fail with an undetectable output format" {
  run uv run mt5cli --output "${BATS_TEST_TMPDIR}/out.xyz" account-info
  [[ "${status}" -ne 0 ]]
  [[ ! -e "${BATS_TEST_TMPDIR}/out.xyz" ]]
}
