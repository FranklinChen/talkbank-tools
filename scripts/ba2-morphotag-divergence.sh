#!/usr/bin/env bash
# Report how BA3's morphotag output diverges from Batchalign 2's reference
# outputs, with `batchalign3 compare-runs morphotag`.
#
# A REPORT, not a gate. BA2's January 2026 outputs
# (batchalign/tests/golden/ba2_reference/, commit 84ad500b) are a point of
# comparison, not a specification: BA3 deliberately departs from them (UD
# root convention, no invented %mor features, its own L2 merge). Every
# difference is evidence for a reader to classify; nothing here passes or
# fails on a difference.
#
# Usage:
#   scripts/ba2-morphotag-divergence.sh OUTPUT_DIR
#
# OUTPUT_DIR must not exist. The script morphotags the parity fixtures
# (batchalign/tests/support/parity/) that have a BA2 reference, twice: with
# the default tokenization against ba2_reference/morphotag/, and with
# --retokenize against ba2_reference/morphotag_retok/. It then writes, per
# mode, the two run manifests, a comparison plan, and the compare-runs
# report under OUTPUT_DIR/MODE/report/runs/COMPARISON_ID/ (report.json,
# summary.csv, pairs/, review/). Fixtures BA3 produced no output for
# (refused as invalid CHAT, or failed) are listed one per line in
# OUTPUT_DIR/MODE/no-output.txt, each with the first error `chatter
# validate` reports for it, and left out of the comparison.
#
# Three parity fixtures are invalid CHAT and stay so: eng_bilingual
# (`and [- spa]` mid-utterance), eng_complex_tiers (a bare `0_1000` bullet,
# `@Options: multi` out of place) and spa_clinical (`[=! ...]` before any
# word). BA2's references were generated from them as they are, and BA2 is
# not rerun, so a corrected fixture would be compared with the output of a
# different input. BA3 refuses them at pre-validation; no-output.txt says
# why.
#
# The batchalign3 on PATH runs the morphotag jobs, through its managed
# server as usual; set BATCHALIGN3 to use another binary. The report names
# that binary's build identity.

set -euo pipefail

usage() {
    echo "usage: $0 OUTPUT_DIR" >&2
    exit 64
}

[[ $# -eq 1 ]] || usage
out=$1
if [[ -e "$out" ]]; then
    echo "error: $out exists; give a new directory" >&2
    exit 64
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${BATCHALIGN3:-batchalign3}
fixtures=$repo/batchalign/tests/support/parity
reference=$repo/batchalign/tests/golden/ba2_reference
# The BA2 baseline every reference file was generated with
# (batchalign/tests/support/PROVENANCE.md).
ba2_build=84ad500b

# `batchalign3 version` writes its identity to stderr. A failing binary is
# reported with what it printed, not captured silently into the identity.
version_status=0
build=$("$bin" version 2>&1) || version_status=$?
if [[ $version_status -ne 0 || -z "$build" ]]; then
    echo "error: $bin version failed (exit $version_status): $build" >&2
    exit 1
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)

# The first error `chatter validate` reports for FILE, or why there is none.
first_validation_error() {
    local file=$1 report first
    if ! command -v chatter >/dev/null 2>&1; then
        echo "not validated: chatter is not on PATH"
        return
    fi
    # chatter exits non-zero for an invalid file, with the errors as its
    # report; any other failure (a crash, an unreadable file) is reported as
    # that, never as a valid file.
    local status=0
    report=$(chatter validate "$file" 2>&1) || status=$?
    first=$(grep -m1 -E 'error\[E[0-9]+\]' <<<"$report" || true)
    if [[ $status -eq 0 ]]; then
        echo "valid CHAT: the morphotag job failed for another reason (see its log)"
    elif [[ -n "$first" ]]; then
        echo "$first"
    else
        echo "not validated: chatter validate failed (exit $status): $(head -n 1 <<<"$report")"
    fi
}

# Morphotag the fixtures that have a reference in REFERENCE_DIR, and report
# the divergence. MODE names the run; FLAGS are extra morphotag arguments.
report_mode() {
    local mode=$1 reference_dir=$2
    shift 2
    local work=$out/$mode
    local input=$work/input ba3=$work/ba3 plan=$work/comparison.toml
    mkdir -p "$input"

    local ref name
    local -a names=()
    shopt -s nullglob
    local -a refs=("$reference_dir"/*.jan9.cha)
    shopt -u nullglob
    if [[ ${#refs[@]} -eq 0 ]]; then
        echo "error: no *.jan9.cha references in $reference_dir" >&2
        exit 1
    fi
    for ref in "${refs[@]}"; do
        name=$(basename "$ref" .jan9.cha)
        if [[ ! -f "$fixtures/$name.cha" ]]; then
            echo "error: no parity fixture for reference $ref" >&2
            exit 1
        fi
        cp "$fixtures/$name.cha" "$input/"
        names+=("$name")
    done

    # A fixture with no output (BA3 refuses invalid CHAT at pre-validation,
    # or the file failed) makes the job exit non-zero. The morphotag log
    # above says which and why; the report covers the rest.
    local morphotag_status=0
    "$bin" morphotag "$@" -o "$ba3" "$input" || morphotag_status=$?

    local -a paired=() missing=()
    for name in "${names[@]}"; do
        if [[ -f "$ba3/$name.cha" ]]; then
            paired+=("$name")
        else
            missing+=("$name")
        fi
    done
    if [[ ${#paired[@]} -eq 0 ]]; then
        echo "error: $mode: morphotag produced no output (exit $morphotag_status)" >&2
        exit 1
    fi

    local source_id=ba3-parity-fixtures-$mode
    "$bin" compare-runs manifest machine \
        --artifacts "$ba3" --output "$work/ba3.manifest.json" \
        --run-id "ba3-$mode" --source-id "$source_id" \
        --implementation batchalign3 --command morphotag --build "$build" \
        --argument "tokenization=$mode"
    # compare-runs resolves artifact roots relative to the plan, so the
    # references are copied beside it; the manifest's BLAKE3 hashes tie the
    # copies to the committed files.
    cp -R "$reference_dir" "$work/ba2"
    "$bin" compare-runs manifest machine \
        --artifacts "$work/ba2" --output "$work/ba2.manifest.json" \
        --run-id "ba2-jan9-$mode" --source-id "$source_id" \
        --implementation batchalign2 --command morphotag --build "$ba2_build" \
        --argument "tokenization=$mode"

    {
        echo 'schema_version = 1'
        echo 'pairing = "same_source_chat"'
        echo 'output = "report"'
        echo
        echo '[left]'
        echo 'manifest = "ba3.manifest.json"'
        echo 'artifacts = "ba3"'
        echo
        echo '[right]'
        echo 'manifest = "ba2.manifest.json"'
        echo 'artifacts = "ba2"'
        for name in "${paired[@]}"; do
            echo
            echo '[[pairs]]'
            echo "left = \"$name.cha\""
            echo "right = \"$name.jan9.cha\""
        done
    } > "$plan"

    # compare-runs exits non-zero both when it refuses the plan and when a
    # pair could not be compared after the report was written. Only the
    # second is a finding to read in the report.
    local status=0
    "$bin" compare-runs morphotag --plan "$plan" || status=$?
    if [[ ! -d "$work/report/runs" ]]; then
        echo "error: $mode: compare-runs wrote no report (exit $status)" >&2
        exit 1
    fi
    if [[ $status -ne 0 ]]; then
        echo "note: $mode: some pairs could not be compared; see the report" >&2
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        for name in "${missing[@]}"; do
            printf '%s: %s\n' "$name" "$(first_validation_error "$input/$name.cha")"
        done > "$work/no-output.txt"
        echo "$mode: no BA3 output, not compared (reasons in $work/no-output.txt): ${missing[*]}"
    fi
    echo "$mode: report under $work/report/runs/"
}

report_mode keeptokens "$reference/morphotag"
report_mode retokenize "$reference/morphotag_retok" --retokenize
