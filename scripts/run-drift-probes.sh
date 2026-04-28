#!/usr/bin/env bash
# run-drift-probes.sh — execute the Stanza drift probes and emit a
# report (never fails the caller).
#
# Phase G of the test-cost revamp. Probes (@pytest.mark.mwt_probe
# and @pytest.mark.decision_probe) are monitors that pin Stanza's
# current tokenization + normalization behavior. A probe "failure"
# is drift worth adjudicating at leisure — not a push-blocker.
#
# This wrapper:
#   1. Points the Phase B history writer at a dedicated probe-history
#      SQLite so probe outcomes don't mingle with regular test history.
#   2. Runs the probe suite via pytest (with golden RAM guard off by
#      default — caller can override).
#   3. Generates a markdown report at
#      docs/investigations/drift-<date>-stanza.md via
#      scripts/generate_drift_report.py.
#   4. Exits 0 unconditionally — reports are the deliverable.
#
# Usage:
#   bash scripts/run-drift-probes.sh                    # mwt + decision
#   bash scripts/run-drift-probes.sh --mwt-only         # just mwt_probe
#   bash scripts/run-drift-probes.sh --decision-only    # just decision_probe

set -euo pipefail

DB="${BATCHALIGN_DRIFT_DB:-${HOME}/.batchalign3/drift-probe-history.sqlite}"
# Reports land in the private workspace's investigations directory.
# Resolves relative to the repo root by default (this script assumes
# it's invoked from the batchalign3 checkout). Override with
# BATCHALIGN_DRIFT_REPORT_DIR for a non-standard workspace layout.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="${BATCHALIGN_DRIFT_REPORT_DIR:-${REPO_ROOT}/../docs/investigations}"
SELECT='golden and (mwt_probe or decision_probe)'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mwt-only) SELECT='golden and mwt_probe'; shift ;;
        --decision-only) SELECT='golden and decision_probe'; shift ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "error: unknown arg $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$(dirname "$DB")"

echo "== drift probes: writing to $DB"
echo "== selection: $SELECT"

# Run the probes. We intentionally do NOT `set -e` over this line —
# a non-zero exit from pytest (drift = failures) is the expected
# case when something interesting happened.
BATCHALIGN_TEST_HISTORY_DB="$DB" \
    uv run pytest -q --no-header -m "$SELECT" \
    -o addopts= -p no:cacheprovider \
    || true

echo "== generating report"
uv run python scripts/generate_drift_report.py \
    --db "$DB" \
    --output-dir "$REPORT_DIR" \
    --window-days 1
