#!/usr/bin/env bash
# test-bg-status.sh — list running + recent test-bg runs.
#
# Reads ~/.batchalign3/bg-test/*/ and reports: which runs are still
# alive (process exists AND no .status file yet), which have
# finished, and what their exit codes were.

set -euo pipefail

ROOT="${HOME}/.batchalign3/bg-test"
LIMIT="${1:-10}"

if [[ ! -d "$ROOT" ]]; then
    echo "(no runs — $ROOT does not exist)"
    exit 0
fi

# Enumerate meta files, newest first, capped at LIMIT entries.
shopt -s nullglob
metas=( "$ROOT"/*/*.meta )
shopt -u nullglob

if [[ ${#metas[@]} -eq 0 ]]; then
    echo "(no runs under $ROOT)"
    exit 0
fi

# Sort by basename timestamp (descending). Rely on the <ts>.meta
# naming — bash doesn't have a portable array-sort, so delegate.
IFS=$'\n' sorted=( $(printf '%s\n' "${metas[@]}" | awk -F/ '{print $NF"\t"$0}' | sort -t. -k1,1nr | cut -f2-) )
unset IFS

printf '%-28s  %-8s  %-8s  %-8s  %s\n' 'SLUG/TS' 'STATE' 'EXIT' 'DUR(s)' 'LOG'

count=0
for meta in "${sorted[@]}"; do
    (( count++ )) || true
    if (( count > LIMIT )); then break; fi

    ts_base="$(basename "$meta" .meta)"
    slug="$(basename "$(dirname "$meta")")"
    run_dir="$(dirname "$meta")"
    status_file="${run_dir}/${ts_base}.status"
    log_file="${run_dir}/${ts_base}.log"

    pid=""
    duration=""
    exit_code=""
    if [[ -f "$meta" ]]; then
        # shellcheck disable=SC1090
        pid="$(grep -E '^pid=' "$meta" | tail -1 | cut -d= -f2- || true)"
        duration="$(grep -E '^duration_s=' "$meta" | tail -1 | cut -d= -f2- || true)"
    fi

    if [[ -f "$status_file" ]]; then
        exit_code="$(cat "$status_file" 2>/dev/null || echo '?')"
        if [[ "$exit_code" == "0" ]]; then
            state="DONE-OK"
        else
            state="DONE-FAIL"
        fi
    elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        state="RUNNING"
    else
        state="ORPHAN"
    fi

    printf '%-28s  %-8s  %-8s  %-8s  %s\n' \
        "${slug}/${ts_base}" "$state" "${exit_code:-—}" "${duration:-—}" "$log_file"
done
