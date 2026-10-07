# morphotag: Developer Reference

**Status:** Current
**Last updated:** 2026-10-03 08:50 EDT

Implementation guide for the `morphotag` command. For user-facing
documentation, see [User Guide: morphotag](../../user-guide/commands/morphotag.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `MorphotagArgs`, `MorphotagPolicyArgs` | I/O, tokenization, multilingual handling, lexicon, and analysis policy (`--no-l2-morphotag`, `--no-pos-hints`, `--ca-policy`) |
| Options builder | `crates/batchalign/src/cli/args/options.rs` (inline dispatch) | Maps CLI values to wire-compatible `MorphotagOptions` |
| Job dispatch | `crates/batchalign/src/execution/morphotag/mod.rs` | Bounded per-file fanout and durable writeback |
| Runtime policy | `crates/batchalign/src/types/params.rs`: `MorphotagExecutionPolicy` | Lowers wire booleans into typed L2, `$POS`, and CA policies |
| Input admission and CA policy | `crates/batchalign/src/pipeline/morphosyntax/states.rs`: `ParsedFile` | Consumes Chatter's header-aware source admission into whole-file pass-through or admitted analysis |
| Morphosyntax orchestration | `crates/batchalign/src/morphosyntax/` | Per-file collection, multilingual worker dispatch, result injection |
| Worker boundary | `crates/batchalign/src/execution/worker_gateway.rs` | Delegates each file to the consuming pipeline, without a separate language parse |
| Injection | `talkbank-transform::morphosyntax` | Writes `%mor`/`%gra` from matched typed UD annotations |
| Retokenization | `crates/batchalign/src/retokenize/` | Character-level DP for Stanza word splits/merges |
| Payload collection & injection | `crates/batchalign-transform/src/morphosyntax/`: `collect_payloads()`, `clear_morphosyntax()`, `inject_results()`, `remove_empty_morphosyntax_placeholders()` | Cross-crate: domain logic lives in talkbank-transform model layer |
| Worker IPC | `batchalign/inference/morphosyntax.py`: `batch_infer_morphosyntax()` | Loads Stanza, returns raw `to_dict()` UD annotations |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`:
the CLI posts source/output path lists instead of CHAT bytes. See
[Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## Single-file phase ownership

The single-file path in `pipeline/morphosyntax.rs` uses consuming transitions:

```mermaid
flowchart LR
    P["One source-bound parse and header plan"] --> CA["Fully admitted CA pass-through"]
    CA --> CS["Strip legacy decision tiers and serialize"]
    P --> A["MOR/GRA removed; all retained CHAT admitted"]
    A --> B["Collected payloads and hint evidence"]
    B --> I["Inferred: matching responses or NoWork"]
    I --> R["Applied morphology"]
    R --> K["PostChecked"]
    K --> S["Provenance, placeholder cleanup, serialization"]
```

`Analysis<S>` owns the CHAT document and its per-file language. Each phase
exposes only the next operation: an unparsed or unadmitted document cannot
reach payload collection, and injection requires an inferred phase carrying
its payloads and matching response count. `HintPlan` contains captured evidence
when requested; there is no separate flag permitting a missing evidence value.
The shared transform boundary now owns `MatchedMorphosyntaxResponses`:
construction rejects missing or extra utterance responses, and binding to the
mutable CHAT document checks every destination index before any injection.
The existing `inject_results` entry point admits through the same type, while
the typed pipeline carries the admitted batch directly into its consuming
`inject` method. This prevents `zip` truncation and stale-index panics. Batch
admission is distinct from the existing per-token linguistic mismatch policy,
which still records diagnostics. Equal counts and valid indices do not prove
response ordering or linguistic correctness.

Job-level language is excluded from the immutable run options. The two language
representations needed by the worker and model APIs are resolved once.

CA pass-through remains a separate policy outcome. It does not claim analysis
admission, resolve an inference language, or traverse six no-op analysis stages.
It does require complete retained-input admission. `PostChecked` means the
existing output gate and command-specific completion checks passed; it is
distinct from the complete input proof. Neither branch stores optional final
output. See [input admission ownership](../../architecture/morphotag-invariants.md#input-admission-and-replacement-ownership).

The shared generic `observe_stage` helper retains the existing start/completion
trace fields and duration measurement for transitions that execute. It accepts
and returns the transition's concrete types without boxing futures or building
an eight-stage dependency graph for every file. Other command pipelines still
use the dynamic planner and share the same observation helper.

---

## Per-file scheduling and inference batching

The dispatcher fans files out through the job's memory-aware worker limit and
writes each completed file independently. Each file's admitted utterances are
collected and batched for inference through the shared managed worker pool.
It does not pool the entire corpus into a single cross-file inference request.

---

## Repeated and incremental runs

Morphotag does not use the persistent audio-task cache. A repeated full run
sends every applicable utterance through Stanza again. To preserve existing
`%mor` and `%gra` on unchanged utterances, pass the prior CHAT tree with
`--before`; the incremental path diffs typed utterances and dispatches only
inserted or word-changed material.

---

## Worker IPC: morphosyntax task

```text
batch_infer request:
{
  "task": "morphosyntax",
  "items": [
    { "words": ["hello", "world"], "lang": "eng",
      "terminator": ".", "special_forms": [] },
    ...
  ]
}

batch_infer response:
[
  [
    { "id": [1], "text": "hello", "upos": "INTJ", "lemma": "hello",
      "head": 2, "deprel": "discourse", "feats": {} },
    ...
  ],
  ...
]
```

The Rust injection layer (`inject_results()`) maps the UD annotation back to
the CHAT AST word by word, writing `%mor` (`pos|lemma` notation) and `%gra`
(`idx head deprel`) tiers.

### Upstream-defect ingress filter (Python side)

Before the UD annotations cross back from Python to Rust,
`batch_infer_morphosyntax` runs a known-defect workaround over every
Stanza sentence: any ``<SOS>``, ``<EOS>``, ``<UNK>``, ``<PAD>``, ``<s>``,
``</s>`` (and similar neural-LM control-token) substrings that leaked
into ``word.text`` or ``word.lemma`` are stripped in place. Every
rewrite emits a ``tracing.warning`` naming the language, the leaked
value, and the post-strip replacement so the workaround is visible
in ``~/.batchalign3/server.log``.

This is Defect 4 in the [Stanza Limitations
registry](../../reference/stanza-limitations.md#defect-4-neural-lm-control-tokens-leak-into-document-output-finnish-mwt).
The minimum trigger is 3+-word Finnish input containing ``tollei`` on
Stanza 1.11.1. ``chatter validate`` is the downstream gate if a leak
variant escapes the filter's vocabulary.

Code + tests:

- `batchalign/inference/_control_token_filter.py`: pure stripper + regex
- `batchalign/inference/morphosyntax.py`: call site inside
  `batch_infer_morphosyntax` after `doc.to_dict()`
- `batchalign/tests/inference/test_control_token_filter.py`,
  34 pure-function tests (regex vocabulary, strip contract, MWT safety)
- `batchalign/tests/pipelines/morphosyntax/test_stanza_fi_mwt_sos_leak.py`
 , standalone upstream reproducer
- `batchalign/tests/pipelines/morphosyntax/test_control_token_leak_propagation.py`
 , integration test through `batch_infer_morphosyntax`

The five-step workflow that produced this filter is the same one any
future upstream defect should follow, see
[Upstream Defect Policy](../upstream-defect-policy.md).

### Language-group failure propagation (Rust side)

A separate layer at the Rust orchestrator handles **pool-level** failures:
worker saturation, timeouts, worker crashes, and the now-retired
"would deadlock" bailout (replaced by idle-cross-group
eviction in `worker/pool/eviction.rs`). The typed
[`LanguageGroupFailure`](https://github.com/FranklinChen/talkbank-tools/blob/main/crates/batchalign/src/morphosyntax/outcomes.rs)
aggregator receives per-group outcomes, and
[`classify_file_for_injection`](https://github.com/FranklinChen/talkbank-tools/blob/main/crates/batchalign/src/morphosyntax/outcomes.rs)
routes every file whose utterance range intersects a failed group to
`TextBatchFileResult::err`, skipping `inject_results()` so no CHAT is
serialized with stripped tiers.

The two layers are complementary:

| Layer | Handles | Response |
|---|---|---|
| Python ingress filter | Upstream library produces bad individual tokens | Strip + log; file lands clean |
| Rust failure propagation | Pool-level failure produced no response at all | Per-file error; file not written |

Code: `crates/batchalign/src/morphosyntax/outcomes.rs` (pure
aggregator + classifier, 13 tests),
`crates/batchalign/src/morphosyntax/dispatcher.rs` (trait boundary
for fake-pool tests),
`crates/batchalign/src/morphosyntax/saturation_tests.rs` (3
end-to-end corruption-regression tests).

See also: [Batchalign Workers, Saturation Safeguards](../../../architecture/runtime/batchalign-workers.md#worker-pool-saturation-safeguards).

---

## Pipeline stages: parse → clear → collect → infer → inject → serialize

Morphotag runs a re-entrant pipeline that must preserve CHAT round-trip
fidelity. The important stages in order:

```mermaid
flowchart TD
    parse["Parse CHAT\n(batchalign, tree-sitter)"]
    clear["clear_morphosyntax()\n(morphosyntax/payload.rs)\nreplace Mor/Gra in place with EMPTY\nMorTier::new_mor(Vec::new()) / GraTier::new_gra(Vec::new())"]
    collect["collect_payloads() + collect_pos_hints()\nCapture typed $POS evidence before retokenization\nhas_mor requires Mor tier to be NON-EMPTY"]
    infer["Batch infer\n(runner/dispatch/infer_batched.rs)"]
    inject["inject_results()\n(morphosyntax/injection.rs)\nwrites %mor and %gra into the SAME slots"]
    l2["L2 dispatch (if enabled)\n(morphosyntax/l2/*)"]
    poshints["Apply captured POS evidence (if enabled)\n(morphosyntax/pos_hints.rs)"]
    sweep["remove_empty_morphosyntax_placeholders()\n(morphosyntax/payload.rs)\nserialize-time sweep"]
    serialize["Serialize CHAT\n(talkbank-transform)"]

    parse --> clear --> collect --> infer --> inject --> l2 --> poshints --> sweep --> serialize

    clear -. "preserves tier position in dependent_tiers" .-> inject
    collect -. "skips only utterances whose Mor tier is non-empty" .-> infer
    collect -. "typed hint evidence survives main-tier retokenization" .-> poshints
```

Diagram verified against: `crates/batchalign/src/morphosyntax/batch.rs` (orchestration),
`crates/batchalign-transform/src/morphosyntax/injection.rs` (inject_results),
`crates/batchalign-transform/src/morphosyntax/payload.rs` (clear/collect/sweep),
`crates/batchalign-transform/src/morphosyntax/pos_hints.rs` (capture/apply),
`crates/batchalign/src/chat_ops/morphosyntax_ops/tests.rs` (tier-order regression tests).

### Tier-order preservation

`clear_morphosyntax` previously removed the `%mor` and `%gra` dependent
tiers outright. `inject_results` then called the old "remove-then-add"
pattern at the end of `dependent_tiers`, so regenerated tiers were
displaced to the tail of the list. On files whose source layout put
`%wor` last, very common, the round trip `parse → clear → infer →
inject → serialize` produced a large, spurious tier-order diff.

The current pattern:

1. `clear_morphosyntax` **replaces** the Mor/Gra entries in place with
   empty `MorTier::new_mor(Vec::new())` and `GraTier::new_gra(Vec::new())`.
   Original tier position is retained.
2. `inject_results` writes into the same slots.
3. `remove_empty_morphosyntax_placeholders` is called at serialize time
   to remove any still-empty placeholders (utterances where no
   morphosyntax was produced).
4. `crates/batchalign/src/chat_ops/fa/mod.rs::add_wor_tier` (line 244) uses
   `replace_or_add_tier` instead of the old `remove_wor_tier + push`
   sequence, applying the same preservation principle to `%wor`.

**Regression tests** (in `crates/batchalign/src/chat_ops/morphosyntax_ops/tests.rs`):

- `clear_then_reinject_preserves_tier_order_mor_gra_wor` (line 1185)
- `add_wor_tier_preserves_tier_order_wor_mor_gra` (line 1258)
- `collect_payloads_treats_empty_mor_placeholder_as_unprocessed` (line 1302), tier-order test covering the empty-placeholder sweep

### `collect_payloads` empty-placeholder fix

`collect_payloads` uses a `has_mor` check to skip utterances that have
already been annotated. The original check tested only for the presence
of the `DependentTier::Mor` variant, which meant the empty placeholders
left by `clear_morphosyntax` looked "already processed" and every
utterance was skipped.

Net effect before the fix: `collect_payloads` returned zero payloads
after clearing, the worker was never called, and `%mor` / `%gra` were
silently stripped from the entire file.

`has_mor` now requires the Mor tier to be **non-empty**: it returns
false for an empty placeholder. Regression test:
`collect_payloads_treats_empty_mor_placeholder_as_unprocessed`.

This is a single-call, four-regression-test cluster (three for
tier-order preservation plus this one) that together pin the round-trip
contract.

---

## Pre-validation gate

The authoritative contract is [input admission and replacement
ownership](../../architecture/morphotag-invariants.md#input-admission-and-replacement-ownership):
all retained CHAT must be valid, not merely Level 2. CA pass-through cannot
claim a regeneration exemption, and incremental tier reuse requires full
admission of the before-file.

---

## Language-group concurrency control

Multilingual CHAT files produce batch items with different per-item languages.
Each language group must be dispatched to a worker loaded with the correct Stanza
model, sending French text to an English MWT pipeline produces corrupt Range tokens.

Language groups are dispatched **concurrently** using a semaphore to prevent deadlock:
each language group acquires a semaphore permit before accessing the worker pool.
This ensures that we never try to start more language groups simultaneously than
the worker pool can support (`max_total_workers / max_workers_per_key`).

When a language group finishes and releases its permit, the next waiting group
acquires it and starts, no deadlock, full utilization, all groups eventually
complete. This is the same concurrency pattern the FA pipeline uses for per-file
parallelism.

Implementation: `crates/batchalign/src/morphosyntax/batch.rs:288-312`.

---

## Retokenization (`--retokenize`)

When `--retokenize` is set, `TokenizationMode::StanzaRetokenize` is passed to
the worker. Stanza may split or merge words on the main tier to match UD
tokenization. The retokenization is implemented in Rust
(`crates/batchalign/src/retokenize/`) using a character-level
Hirschberg DP to map Stanza tokens back to original CHAT positions. Existing
`%wor` timing bullets become stale after retokenization.

---

## Per-utterance language routing

Individual utterances with `[- lang]` precodes are routed to the Stanza
pipeline for that language, regardless of the file-level `@Languages` header.
The routing table is built by `collect_payloads()` in the morphosyntax
orchestration layer.

See [Language Routing](../../../architecture/language-and-multilingual/language-routing.md#per-utterance-routing-into-stanza).

---

## L2 morphotag dispatch (default; opt out via `--no-l2-morphotag`)

By default the morphotag pipeline defers the legacy `L2|xxx`
blanking for `@s` words and instead routes them to a
secondary-language Stanza model, merges the response with the
primary model's structural analysis, and splices the merged result
back into the CHAT AST. `--no-l2-morphotag` restores the legacy
blanking behavior.

```mermaid
sequenceDiagram
    participant Orch as Orchestrator<br/>(morphosyntax/batch.rs)
    participant Primary as Primary Stanza<br/>(e.g. deu)
    participant L2 as L2 extractor<br/>(l2/extract.rs)
    participant Spans as Span planner<br/>(l2/plan.rs)
    participant Secondary as Secondary Stanza<br/>(e.g. eng)
    participant Merge as Merge algorithm<br/>(l2/merge.rs)
    participant Splice as Splice<br/>(l2/splice.rs)

    Orch->>Primary: Full utterance,<br/>L2 blanking deferred
    Primary-->>Orch: UD annotations for all words<br/>including @s words
    Orch->>L2: inject primary results
    L2-->>Orch: InjectionResult::l2<br/>(positions + unaligned utterances)
    Orch->>Spans: plan_dispatch_spans(positions)
    Spans-->>Orch: Vec&lt;L2SpanPlan&gt;<br/>(contiguous same-lang, own their positions)
    loop For each target language
        Orch->>Secondary: infer_batch(retokenize=true)<br/>one sentence per span
        Secondary-->>Orch: Vec&lt;UdResponse&gt;
        loop For each span
            Orch->>Merge: merge_planned_secondary_span(span, sentence)
            Note over Merge: secondary owns category, lemma,<br/>features, in-span relations;<br/>external relation checked against<br/>the secondary root's category
            Merge-->>Orch: MergedL2Span
        end
    end
    Orch->>Splice: splice_l2_into_chat(merged spans)
    Splice-->>Orch: SpliceOutcome<br/>(spliced / fallback / gra_upgraded)
```

**Module layout** (`crates/batchalign-transform/src/morphosyntax/l2/`):

| Module | Responsibility |
|--------|----------------|
| `alignment.rs` | `UdAlignment`, the one CHAT-word to UD-word alignment per sentence, lookups by UD id, typed refusals. |
| `extract.rs` | Aligns each primary sentence with a dispatchable `@s` word and records the primary's relation, head and dependents for each `@s` word; reports unaligned utterances. |
| `plan.rs` | Groups deferred positions into contiguous same-language spans that own them, and decides each span's `L2Attachment`. |
| `merge.rs` | Merges a span with its secondary analysis into a `MergedL2Span`: the secondary's items and in-span relations, PART for a phrasal particle, the external relation checked against the secondary root's category. |
| `deprel.rs` | `UdDeprel` newtype, relation-to-category check, relation inference from a category. |
| `splice.rs` | Replaces `L2\|xxx` with each merged span in the CHAT AST, validates, rolls back. |
| `crates/batchalign/src/morphosyntax/batch.rs` | Thin adapter that submits the planned secondary spans to workers and hands the results back to the transform-layer seam. |

**Dispatch wiring:** `crates/batchalign/src/morphosyntax/batch.rs::dispatch_secondary_l2`.
The secondary sentence is aligned to the span's words, so every span word
gets its sentence context (and phrasal-particle recognition) whatever
extra rows, such as the terminator, the sentence carries. A sentence that
does not align or map is an `L2MergeError` for the span, logged, and its
words stay `L2|xxx`.

See [L2 Morphotag: Per-Word Code-Switching Analysis](../../reference/l2-morphotag.md)
for the design rationale and merge algorithm details.

---

## Validation and normalization policy for `@s`

- E255 is now the hard-stop policy for whole-utterance same-language all-`@s`
  patterns. Morphotag does not auto-normalize those utterances; the transcript
  must use `[- lang]`.
- E254 is warn-only for explicit `@s:LANG` markers whose `LANG` is absent from
  `@Languages`. Dispatch still uses the explicit target language; the warning is
  about header drift, not routing failure.
- `chatter debug fix-s` is the companion repair tool. It rewrites qualifying
  whole-utterance `@s` runs to `[- lang]`, appends missing explicit languages to
  `@Languages`, and skips files that are already correct.

---

## Transcriber `$POS` hint post-pass (enabled by default)

After injection completes, if `--respect-pos-hints` is enabled (default; opt-out
via `--no-pos-hints`), the morphotag pipeline walks the ChatFile and overrides
`%mor` POS categories that disagree with transcriber `$POS` annotations.

The post-pass preserves:
- Lemma (from Stanza)
- Morphological features (from Stanza)

Only the POS category (UPOS → CLAN notation) is checked and potentially overridden.
Lemma and features from Stanza remain intact.

The outcome tracks 5 categories:
- `hints_considered`: total `$POS` annotations found
- `hints_agreed`: transcriber POS matched Stanza output (no change needed)
- `hints_overridden`: Stanza POS overridden by transcriber annotation
- `hints_unmapped`: transcriber POS code not in the CLAN↔UPOS mapping table
- `hints_skipped_no_mor`: utterance had no `%mor` tier to override

Implementation: `crates/batchalign/src/chat_ops/morphosyntax_ops/pos_hints.rs`,
`batchalign/src/morphosyntax/batch.rs:509-521`.

---

## Morphosyntax alignment validation

After injection, the pipeline runs alignment validation to detect `%mor`/`%gra`
sync issues. The validator checks:
- Each `%mor` tier has a corresponding `%gra` tier
- Word counts match
- Dependency head indices are in bounds

Validation errors are **warnings only** (logged but non-fatal). Files are still
serialized so invalid files can be inspected for debugging. The validation gate
exists to catch corruption from upstream defects (e.g., Stanza control-token
leaks) that escape the ingress filter.

Implementation: post-injection alignment validation lives in
`crates/batchalign-transform/src/morphosyntax/injection.rs` (the module
top comment names the responsibility, and the typed
`MisalignmentBug` / `MisalignmentDiagnostic` paths fire at the result
sites in that file). The batchalign-side `morphosyntax/worker.rs`
calls the injection path and propagates the validation outcomes.

```bash
# Unit tests (no ML models)
make test

# Morphotag golden tests (real Stanza models, only on Fleet/Large-tier hosts with the models present)
cargo test -p batchalign --features ml-golden --test ml_golden morphosyntax::

# Retokenization unit tests
cargo test -p batchalign retokenize::
```

---

## Related developer documentation

- [Command Flowcharts: morphotag](../../architecture/command-flowcharts.md#morphotag), detailed runtime flowchart
- [Morphosyntax Pipeline](../../reference/morphosyntax.md), %mor/%gra format
- [Stanza Capability Registry](../../architecture/stanza-capability-registry.md)
- [Incremental Processing](../../architecture/incremental-processing.md), `--before` flag
- [Adding Commands](../adding-commands.md), use `morphotag` as the reference for `CrossFileBatchTransform`
