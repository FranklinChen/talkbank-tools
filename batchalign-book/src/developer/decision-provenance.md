# Decision Provenance

**Status:** Current
**Last updated:** 2026-03-30 22:06 EDT

## Overview

Every batchalign3 command makes decisions that alter output: clamping
timestamps, stripping timing, skipping utterances, defaulting values,
normalizing text. These decisions were previously logged via `tracing` but
invisible to the user in the output CHAT file.

The **decision provenance system** surfaces these decisions as `%xalign` and
`%xrev` dependent tiers so users can review what the pipeline did and why.

## Architecture

**Single source of truth:** Every decision point creates a `DecisionRecord`
via `new_and_trace()`, which emits a structured tracing event AND returns
the record. No duplicate logging — the `DecisionRecord` is the source of
truth; tracing events, `%xalign` tiers, and dashboard warnings are all
derived from it.

```mermaid
flowchart TD
    point["Decision point\n(clamping, skipping, defaulting)"]
    record["DecisionRecord::new_and_trace()"]
    trace["tracing::warn! / info!\n(terminal + daemon log)"]
    collect["Vec&lt;DecisionRecord&gt;\ncollected by orchestrator"]
    inject["inject_decision_tiers()"]
    xalign["%xalign: module:strategy reason\n(persistent, in output file)"]
    xrev["%xrev: [?]\n(human review marker)"]

    point --> record
    record --> trace
    record --> collect
    collect --> inject
    inject --> xalign
    inject -->|"needs_review=true"| xrev
```

Key design principle: **callers should NOT separately call `tracing::warn!`**
with the same information. `DecisionRecord::new_and_trace()` handles both
the tracing event and the record creation in one call.

## DecisionRecord

**Source:** `crates/batchalign-chat-ops/src/decisions.rs`

```rust
pub struct DecisionRecord {
    pub line_idx: usize,           // which utterance
    pub speaker: String,           // speaker code
    pub module: DecisionModule,    // pipeline stage
    pub strategy: &'static str,    // what was done
    pub reason: String,            // structured key=value detail
    pub needs_review: bool,        // emit %xrev: [?]?
}
```

### DecisionModule

| Module | Pipeline stage |
|--------|---------------|
| `Fa` | Forced alignment (grouping, injection, postprocessing) |
| `Utr` | Utterance timing recovery |
| `Monotonicity` | Monotonicity enforcement (end-time clamping, start-time stripping) |
| `Morphosyntax` | Stanza mapping, retokenization |
| `Utseg` | Utterance segmentation |
| `Build` | CHAT building from ASR output |
| `AsrPostprocess` | Compounds, numbers, Cantonese, retraces |

### Tier format

`%xalign` content is: `module:strategy reason_string`

```
*CHI:   hello world . ⌈15⌉1000_5000⌈15⌉
%xalign:	monotonicity:end_clamped overlap=1200ms prev_end=6200 next_start=5000
```

When `needs_review` is true, a `%xrev: [?]` tier is also added:

```
*CHI:   hello world .
%xalign:	monotonicity:timing_stripped non_monotonic start_ms=2000 previous_start_ms=5000
%xrev:	[?]
```

## Controlling output

The `--review-level` flag controls which decision tiers are emitted:

| Level | Behavior |
|-------|----------|
| `none` | No decision tiers (smallest output) |
| `low-confidence` (default) | Only `needs_review=true` decisions |
| `all` | Every decision + informational `%xalign: no_decisions` on clean utterances |

## Currently instrumented decisions

### Monotonicity enforcement

| Strategy | needs_review | When |
|----------|:------------:|------|
| `timing_stripped` | yes | Utterance start time is before previous utterance's start (E362 violation). All timing removed. |
| `end_clamped` | no | Utterance end time exceeds next utterance's start. End clamped to next start. |

**Source:** `crates/batchalign-chat-ops/src/fa/orchestrate.rs` —
`enforce_monotonicity()`

### UTR unmatched

| Strategy | needs_review | When |
|----------|:------------:|------|
| `utr_unmatched` | yes | Untimed utterance could not be matched to any ASR tokens during timing recovery. |

**Source:** `crates/batchalign-chat-ops/src/fa/utr.rs` — `run_global_utr()`

### FA word timing drop

| Strategy | needs_review | When |
|----------|:------------:|------|
| `words_timing_dropped` | yes | Word-level timing was dropped because clamping to utterance boundary made start >= end. |

**Source:** `crates/batchalign-chat-ops/src/fa/postprocess.rs` —
`postprocess_utterance_timings()`

### FA bullet repair (existing)

Repair decisions from `fa/repair.rs` are converted to `DecisionRecord` via
`From<&RepairDecision>` and included in the same injection pass.

### Morphosyntax (morphotag command)

| Strategy | needs_review | When |
|----------|:------------:|------|
| `mapping_failed` | yes | UD→CHAT conversion failed for this utterance. No %mor/%gra produced. |
| `retokenization_failed` | yes | Stanza tokens could not be mapped back to CHAT words. No %mor/%gra produced. |
| `injection_failed` | yes | MOR word count mismatch (e.g., MWT expansion). No %mor/%gra produced. |
| `nlp_no_sentences` | yes | Stanza returned an empty response for this utterance. No %mor/%gra produced. |

**Source:** `crates/batchalign-chat-ops/src/morphosyntax/inject.rs`

## Audit of uninstrumented silent decisions

These decisions are logged via `tracing` but not yet tracked as
`DecisionRecord`. They are candidates for future instrumentation:

| Decision | File | Priority |
|----------|------|:--------:|
| FA grouping skip (no timing/estimate) | `fa/grouping.rs:53` | Medium |
| Token stitching partial | `fa/alignment.rs:159` | Medium |
| Hardcoded period terminator | `build_chat.rs:482` | Low |
| Default language (eng) | `build_chat.rs:142` | Low |
| Cantonese normalization | `asr_postprocess/cantonese.rs` | Low |
| Compound merging | `asr_postprocess/compounds.rs` | Low |
| Number expansion | `asr_postprocess/num2text.rs` | Low |
| Retrace detection | `asr_postprocess/mod.rs` | Low |
| Stanza terminator mismatch | `retokenize/parse_helpers.rs:105` | Low |
| Retokenization fallback | `morphosyntax/inject.rs:181` | Medium |

## Integration with existing %xalign/%xrev

The decision provenance system generalizes the existing FA bullet repair
review tier infrastructure (`fa/review_tiers.rs`). The original
`inject_review_tiers()` function and `RepairDecision` type are preserved
for backward compatibility. `DecisionRecord` adds the `module` field and
`From<&RepairDecision>` enables seamless conversion.

The FA orchestrator (`fa/mod.rs`) now collects decisions from all stages
(FA postprocessing, bullet repair, monotonicity enforcement) and injects
them in a single pass via `inject_decision_tiers()`.
