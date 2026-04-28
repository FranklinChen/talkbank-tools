#!/usr/bin/env bash
# choose-test-concurrency.sh — compute a safe parallelism for a test
# profile based on available RAM.
#
# Part of Phase A4 of the test-cost revamp (Franklin, 2026-04-23).
# Budget arithmetic, not machine tiers: we leave 40% of RAM for OS +
# editor + browser, then divide the remainder by the profile's peak
# resident-set estimate to get a safe jobs count.
#
# Usage:
#   jobs=$(bash scripts/choose-test-concurrency.sh default)
#   cargo nextest run --profile interactive --test-threads $jobs
#
# Output: a single integer on stdout; caller captures.
#
# Profiles (match batchalign3/.config/nextest.toml test-groups and
# documented worker memory budgets):
#   default  — pure Rust lib/unit tests, no ML models          ~1 GB
#   ml       — loads Stanza + friends into the test process   ~12 GB
#   gpu      — loads Whisper (FA / ASR)                        ~6 GB
#   stress   — concurrent dispatch test, shared state pressure  ~4 GB
#   python   — pytest without golden marker, no models         ~1 GB
#
# The 128 GB hard-refuse guard for golden + xdist in
# batchalign/tests/conftest.py is correctness (OOM prevention) and
# stays — this script handles tuning below that ceiling.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: choose-test-concurrency.sh PROFILE" >&2
    echo "  profiles: default, ml, gpu, stress, python" >&2
    exit 2
fi

profile="$1"

case "$profile" in
    default|python) peak_mb=1024 ;;
    ml)             peak_mb=12288 ;;
    gpu)            peak_mb=6144 ;;
    stress)         peak_mb=4096 ;;
    *)
        echo "error: unknown profile '$profile'" >&2
        exit 2
        ;;
esac

# Total system RAM in MB, macOS-first then Linux fallback.
total_mb=0
if [[ "$(uname -s)" == "Darwin" ]] && command -v sysctl >/dev/null 2>&1; then
    total_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    total_mb=$(( total_bytes / 1024 / 1024 ))
elif [[ -r /proc/meminfo ]]; then
    kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    total_mb=$(( kb / 1024 ))
fi

if [[ "$total_mb" -le 0 ]]; then
    # Unknown host — be conservative, single-threaded.
    echo 1
    exit 0
fi

# 60% of total RAM available to the test process; the other 40% is
# reserved for the rest of the user's environment. This is the
# "budget arithmetic" from the plan.
available_mb=$(( total_mb * 60 / 100 ))
jobs=$(( available_mb / peak_mb ))

# Floor at 1 — never emit 0 (caller would pass --test-threads 0).
if [[ "$jobs" -lt 1 ]]; then
    jobs=1
fi

# Ceiling at CPU count — no benefit from oversubscription. Use
# sysctl on macOS (hw.ncpu) or /proc/cpuinfo on Linux.
ncpu=0
if [[ "$(uname -s)" == "Darwin" ]] && command -v sysctl >/dev/null 2>&1; then
    ncpu="$(sysctl -n hw.ncpu 2>/dev/null || echo 0)"
elif [[ -r /proc/cpuinfo ]]; then
    ncpu="$(grep -c '^processor' /proc/cpuinfo)"
fi

if [[ "$ncpu" -gt 0 && "$jobs" -gt "$ncpu" ]]; then
    jobs="$ncpu"
fi

echo "$jobs"
