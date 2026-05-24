# translate — Developer Reference

**Status:** Current
**Last updated:** 2026-05-23 09:08 EDT

Implementation guide for the `translate` command. For user-facing
documentation, see [User Guide: translate](../../user-guide/commands/translate.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs` — `TranslateArgs` | `--translate-engine` flag (`TranslateEngine` clap enum) + lang override |
| CLI → wire | `crates/batchalign/src/cli/args/options.rs` — `Commands::Translate` arm | Maps `TranslateEngine` → `Option<TranslateEngineName>` on `TranslateOptions` |
| Command definition | `crates/batchalign/src/commands/translate.rs` | `CommandDefinition` impl |
| Translate orchestration | `crates/batchalign/src/translate.rs` | Cross-file batching, cache, `%xtra` injection |
| Batch dispatch | `crates/batchalign/src/runner/dispatch/infer_batched.rs` | Shared with morphotag and utseg |
| Injection | `crates/batchalign/src/translate.rs` | Writes `%xtra:` tiers from translation strings |
| Engine type | `crates/batchalign/src/types/engines.rs` — `TranslateEngineName` | Wire-format enum (`google` / `seamless`), `EngineBackend` impl, `EngineOverrides.translate` field |
| Engine resolution (server) | `crates/batchalign/src/types/options.rs` — `TranslateOptions::effective_translate_engine` | Precedence: shared `--engine-overrides` `{"translate":"..."}` > `--translate-engine` flag > Google default |
| Engine bootstrap | `batchalign/worker/_model_loading/translation.py::load_translation_engine(bootstrap)` | Reads `bootstrap.engine_overrides["translate"]`, dispatches via exhaustive match to `_load_google_translate` or `_load_seamless_translate`. Unknown engine names raise `ValueError` |
| Engine resolution (worker) | `batchalign/worker/_model_loading/translation.py::resolve_translate_engine` | Pure function from `engine_overrides` dict → `TranslationBackend`; default Google |
| Worker IPC | `batchalign/inference/translate.py` — `batch_infer_translate()` | Iterates batch items, calls the resolved `translate_fn(text, src_lang)`, returns `raw_translation` per item. Sleeps 1.5s per item when backend is `GOOGLE` (rate limit). Pre-processing (Chinese space removal) happens in Rust before the call; post-processing in Rust after |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## Cache key structure

Translation cache keys (BLAKE3 hash of):
- Normalized utterance text
- Source language code
- Target language code (always `eng`)

---

## Worker IPC: translate task

```text
batch_infer request:
{
  "task": "translate",
  "items": [
    { "text": "Bonjour le monde.", "src_lang": "fra", "tgt_lang": "eng" },
    ...
  ]
}

batch_infer response:
[ "Hello world.", ... ]
```

---

## Pre-validation gate

`translate` requires CHAT Level 1.

## Idempotency

`inject_translation` (in `talkbank-transform::translate`) calls
`replace_or_add_tier`, which **overwrites** any existing `%xtra` tier on the
utterance. Re-running `translate` on a file that already has `%xtra` tiers
re-translates and replaces them. This diverges from BA2, which guarded
with `if i.translation: continue` and preserved the first translation.

## Engine selection precedence

`TranslateOptions::effective_translate_engine` mirrors
`AlignOptions::effective_fa_engine` and
`BenchmarkOptions::effective_asr_engine`. From highest priority to
lowest:

1. `common.engine_overrides.translate` — set by
   `--engine-overrides '{"translate":"<engine>"}'`.
2. `TranslateOptions.translate_engine: TranslateEngineName` — set by
   `--translate-engine google|seamless`. Defaults to Google via
   `default_translate_engine()`.

There is deliberately no `server.yaml` knob for engine selection.
Translation engine is a policy choice, not a host fact, and policy
belongs at the invocation site (CLI flag or shell alias) — never in
a config file. See the no-config-junk principle in
`book/src/batchalign/user-guide/commands/translate.md`.

The worker pool key includes the resolved translate engine
(`dispatch_engine_overrides_json` always emits a `translate` entry).
Google and Seamless workers are not interchangeable, so they end up
in separate pools.

## BA2 → BA3 migration notes

| Concern | BA2-jan9 | BA3 |
|---------|----------|-----|
| CLI shape | `batchalign translate IN_DIR OUT_DIR` (separate dirs) | `batchalign3 translate <dir-or-file>` (in-place by default) |
| Default engine | `googletrans` (dispatch.py: `"translate": "gtrans"`) | `googletrans`, with explicit per-host opt-in to Seamless via `server.yaml` `default_translate_engine` or `--translate-engine seamless` |
| Concurrency | Sequential per utterance, with `time.sleep(1.5)` on Google | Batched cross-file dispatch, multiple worker groups per language, 1.5s sleep retained per-item on Google only |
| Re-run behavior | Skip already-translated utterances | Overwrite existing `%xtra` |
| Chinese (yue/zho) preprocessing | Inline in `gtrans.py` only; `seamless.py` did NOT strip spaces (BA2 bug) | Uniform `preprocess_for_translate` in Rust, applied before any backend |
| Per-item failure | Aborts the file (single-file CLI invocation) | Marks the affected file as failed with a typed `TextWorkflowFileError::ItemErrors` carrying the engine error(s); other files in the same cross-file batch continue normally. Transient errors at the batch dispatch layer retry; per-item engine failures propagate to file-level failure without retry. |
| Output tier | `%xtra` | `%xtra` (identical) |

**Tier-name clarification.** Neither BA2 nor BA3 produces a `%tra` tier.
Both versions emit `%xtra`. Any other translation-tier name observed in
the wild was not written by Batchalign.

---

## Testing

```bash
make test
cargo nextest run -p batchalign -E 'test(translate::)'
```

---

## Related developer documentation

- [Command Flowcharts: translate](../../architecture/command-flowcharts.md#translate)
