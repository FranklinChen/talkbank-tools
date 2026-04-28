# Debugging Infrastructure

**Last modified:** 2026-03-27 19:47 EDT

## Design Goal

Every worker pipeline failure should be **diagnosable from a single error
message and a dump file**, without requiring ad-hoc probing, injected print
statements, or multiple reproduction attempts. The system is designed so that
an engineer (or an AI assistant reading error logs) can identify the root cause
in one step.

## Architecture: Three Layers

### Layer 1: Enriched Error Messages (always on)

When a worker request fails, the error message itself carries enough context
to identify the problem:

```
Failed to parse raw Stanza output for item 4
  (words: ["euh", "Lisa", "est", "au", "Mexique", ...]):
  sentence 0 word 3: missing field `lemma`.
  Diagnostics: sentence 0 word 3: field 'upos' — field absent
    (keys present: ["end_char", "id", "start_char", "text"]).
    Stanza's processor likely failed silently for this token.
```

This tells you:
- Which batch item failed (item 4)
- What words were sent (so you can reproduce with the worker directly)
- Which word in the Stanza output is broken (word 3)
- What fields are present vs missing
- A likely cause ("Stanza's processor failed silently")

**Implementation:** `morphosyntax/worker.rs` calls `diagnose_parse_failure()`
from `stanza_raw.rs` on the error path and includes the diagnostics in both
the `tracing::warn!` and the user-facing error message.

### Layer 2: Always-On Failure Dumps (`~/.batchalign3/debug/`)

Critical failures write structured JSON dumps to `~/.batchalign3/debug/`
regardless of whether `--debug-dir` was specified:

| Dump file | Trigger | Contents |
|-----------|---------|----------|
| `batch_failure_morphosyntax_{timestamp}.json` | Morphosyntax batch parse failure | All batch items (words + lang), error message |
| `failed_ipc_{timestamp}.json` | Any worker IPC failure (timeout, crash, protocol) | Full request JSON, error type/message, worker PID + label, response fragment |

These dumps are small (word lists, not audio) and the failure rate is low,
so disk impact is negligible. They accumulate until manually cleaned.

**Why always-on:** The cost of NOT having the dump (hours of ad-hoc debugging)
far exceeds the disk cost (~10 KB per failure). The production morphotag incident
(2026-03-27) motivated this: the failure was deterministic but required
manual package patching to capture the payload.

### Layer 3: Opt-In Debug Artifacts (`--debug-dir`)

When `--debug-dir /path/to/dir` is passed (or `BATCHALIGN_DEBUG_DIR` is set),
the `DebugDumper` writes detailed pipeline artifacts:

| Artifact | Pipeline | Contents |
|----------|----------|----------|
| `{stem}_pre_morphosyntax.cha` | Morphotag | CHAT before morphosyntax injection |
| `{stem}_morphosyntax_extracted.json` | Morphotag | Batch items sent to Python |
| `{stem}_morphosyntax_ud_responses.json` | Morphotag | Raw Stanza output |
| `{stem}_utr_input.cha` | Align | CHAT before UTR timing injection |
| `{stem}_utr_tokens.json` | Align | ASR timing tokens |
| `{stem}_fa_input.cha` | Align | CHAT before FA |
| `{stem}_fa_grouping.json` | Align | FA group structure |
| `{stem}_fa_group_{n}.json` | Align | Per-group FA timings |
| `{stem}_asr_response.json` | Transcribe | Raw ASR output |
| `{stem}_post_asr.cha` | Transcribe | CHAT after ASR assembly |

These are **zero-cost when disabled** — `DebugDumper` methods return
immediately without allocation when constructed without a directory.

## Structured Diagnostics (`stanza_raw::diagnose_parse_failure`)

Instead of relying on raw serde deserialization errors ("missing field
`lemma`"), the diagnostics function scans Stanza's raw `to_dict()` output
and produces structured, actionable reports:

```rust
pub struct StanzaWordDiagnostic {
    pub sentence_idx: usize,
    pub word_idx: usize,
    pub field: String,
    pub issue: String,
}
```

Checks performed:
- **Missing required fields:** `text`, `lemma`, `upos`, `deprel`
- **Null values:** Stanza can emit `"lemma": null` when a processor fails
- **MWT Range tokens:** `id: [start, end]` tokens are expected to lack
  annotation fields — the diagnostics skip lemma checks for these
- **`<pad>` sentinels:** Stanza emits `"deprel": "<pad>"` for padding tokens
- **Type mismatches:** `id` as unexpected type, string fields as non-string

Each diagnostic includes a human-readable explanation of the likely cause,
not just the symptom. This is designed for an engineer (or AI) reading the
log to immediately understand what went wrong and where.

## How This Maximizes AI-Assisted Debugging

The debugging infrastructure is specifically designed for the scenario where
an AI assistant (Claude Code or similar) is analyzing a failure:

### 1. Single-Message Root Cause

The enriched error message contains all the information needed to identify the
root cause without any follow-up queries:

```
words: ["euh", "Lisa", "est", "au", ...]
sentence 0 word 3: field 'upos' absent (keys: ["end_char", "id", "start_char", "text"])
```

An AI can immediately determine: "Word 3 is `au`, which in French triggers
MWT expansion (`à` + `le`). The range token has only positional fields.
The English MWT pipeline is being used for French text."

### 2. Machine-Readable Dump Files

The JSON dumps are structured, not log-grepped text. An AI can:
- Parse the dump
- Extract the exact batch items
- Construct a reproduction command
- Compare dumps across machines
- Track which items fail consistently

### 3. Reproducibility Without the Server

The `failed_ipc_{timestamp}.json` dump includes the full request JSON.
An AI can:
1. Read the dump
2. Extract the request
3. Pipe it to `python -m batchalign.worker --task morphosyntax --lang eng`
4. Compare the output to the dump

This is the foundation for the planned `batchalign3 replay` tool.

### 4. Pattern Recognition Across Failures

Because dumps accumulate in `~/.batchalign3/debug/`, an AI can scan all
failure dumps to identify patterns:
- "All failures are on `fra` language items with word `au`"
- "All failures have keys `[end_char, id, start_char, text]` — MWT range tokens"
- "Failures only occur in batches >50 items"

## Auto-Fix Potential

### What Can Be Auto-Fixed Today

The `normalize_word_dict()` function in `stanza_raw.rs` already auto-fixes
several Stanza output issues:

| Issue | Auto-Fix | When |
|-------|----------|------|
| `"lemma": null` | Default to surface text | Always (since 2026-03-27) |
| `"lemma": ""` | Default to surface text | Always |
| `"lemma"` absent | Default to surface text | Non-Range tokens |
| `"upos": null` | Default to `"X"` (unknown) | Always (since 2026-03-27) |
| `"deprel": null` | Default to `"dep"` | Always (since 2026-03-27) |
| `"deprel": "<pad>"` | Replace with `"dep"` | Always |
| Bogus lemma (punct for word) | Replace with surface text | When text has letters |
| `"id": [n]` (single tuple) | Unwrap to `n` | Always |

### What Could Be Auto-Fixed Next

Based on the MWT range token issue discovered on a worker machine:

| Issue | Proposed Auto-Fix | Risk |
|-------|-------------------|------|
| Range token missing annotation fields | Insert defaults (`upos="X"`, `deprel="dep"`, `lemma=""`) | Low — range tokens are display-only in CHAT |
| Non-Range token missing ALL annotation fields | Default all fields from surface text | Medium — may mask a deeper Stanza misconfiguration |

### What Should NOT Be Auto-Fixed

- **Wrong language pipeline used:** If French text is processed by the English
  pipeline, auto-fixing the output hides the routing bug. The correct fix is
  to route to the right pipeline.
- **Worker crash:** A crashed worker needs investigation, not retry masking.
- **Model version mismatch:** If Stanza 2.0 changes output format, patching
  individual fields hides the real problem.

The principle: **auto-fix known engine quirks, but surface routing and
configuration bugs as errors.** `diagnose_parse_failure()` distinguishes
between the two: known quirks (null lemma, pad deprel) produce diagnostics
with "Stanza's processor likely failed silently", while structural problems
(missing `id`, non-object word) produce diagnostics that indicate a deeper
issue.

## Planned Extensions

### `batchalign3 doctor`

Pre-flight diagnostic that validates the worker pipeline on the current
machine. Sends known test inputs through the actual worker and validates
output structure. Catches machine-specific issues (stale models, wrong
pipeline, missing processors) before they become production failures.

### `batchalign3 replay`

Takes a `failed_ipc_{timestamp}.json` dump and replays the exact request
against a fresh worker. Enables:
- Reproduction without the server
- Cross-machine comparison
- Post-fix verification

### Trace Store Integration

The FA pipeline already writes structured traces to an ephemeral trace store
accessible via `GET /jobs/{id}/traces`. The morphosyntax pipeline should
follow the same pattern: collect extraction items, UD responses, and
diagnostics into a `MorphosyntaxTrace` that the dashboard can display.

## File Reference

| File | What |
|------|------|
| `crates/batchalign-chat-ops/src/morphosyntax/stanza_raw.rs` | `normalize_word_dict()`, `diagnose_parse_failure()`, `StanzaWordDiagnostic` |
| `crates/batchalign-app/src/morphosyntax/worker.rs` | Enriched error path with diagnostics |
| `crates/batchalign-app/src/morphosyntax/batch.rs` | Error-path batch dump |
| `crates/batchalign-app/src/runner/debug_dumper.rs` | `DebugDumper` + `dump_morphosyntax_failed_batch()` |
| `crates/batchalign-app/src/worker/handle/mod.rs` | `dump_failed_ipc_request()` for all IPC failures |
