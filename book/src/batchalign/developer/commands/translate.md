# translate: Developer Reference

**Status:** Current
**Last updated:** 2026-09-15 20:20 EDT

Implementation guide for the `translate` command. For user-facing
documentation, see [User Guide: translate](../../user-guide/commands/translate.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `TranslateArgs` | `--translate-engine` flag, parsed straight into `TranslateEngineName` by `engine_selection_parser::<TranslateEngineName>()` |
| CLI → wire | `crates/batchalign/src/cli/args/options.rs`: `Commands::Translate` arm | No mapping: the flag already holds a `TranslateEngineName`. The CLI-private mirror enum and its hand-written match were removed 2026-08-06; see `SelectableEngine` in `types/engines.rs` |
| Catalog entry | `crates/batchalign/src/recipe_runner/catalog.rs` | the `CatalogEntry` for `translate` |
| Stage recipe | `crates/batchalign/src/recipe_runner/recipes.rs` | `TRANSLATE_RECIPE` |
| Translate orchestration | `crates/batchalign/src/translate.rs` | The cross-file text pipeline (the only one; the per-file entry point was deleted with the workflow trait), per-item result admission including the empty-translation refusal, provenance. No result cache |
| Source model and injection | `crates/batchalign-transform/src/translate.rs` | `TranslationSource` (what the speaker produced) and `render()`, its one renderer; `TranslationText` (a translation with something to apply) and `%xtra` injection |
| Job dispatch | `crates/batchalign/src/execution/translate.rs`: `dispatch_translate_job` | Reached from `runner::routing::dispatch_batched_text_command`; one gateway call per file, each with the source language read from that file's `@Languages:` header |
| Engine type | `crates/batchalign/src/types/engines.rs`: `TranslateEngineName` | Wire-format enum (`google` / `seamless` / `nllb` / `tencent` / `aliyun`), `EngineBackend` impl, `EngineOverrides.translate` field |
| Engine resolution (server) | `crates/batchalign/src/types/options.rs`: `TranslateOptions::effective_translate_engine` | Precedence: shared `--engine-overrides` `{"translate":"..."}` > `--translate-engine` flag > Google default |
| Engine bootstrap | `batchalign/worker/_model_loading/translation.py::load_translation_engine(bootstrap)` | Reads `bootstrap.engine_overrides["translate"]`, dispatches via exhaustive match to `_load_google_translate`, `_load_seamless_translate`, `_load_nllb_translate`, `_load_tencent_translate`, or `_load_aliyun_translate`. Unknown engine names raise `ValueError` |
| Engine resolution (worker) | `batchalign/worker/_model_loading/translation.py::resolve_translate_engine` | Pure function from `engine_overrides` dict → `TranslationBackend`; default Google |
| Worker IPC | `batchalign/inference/translate.py`: `batch_infer_translate()` | Iterates batch items through the loaded translation record and returns one tagged item per input: `translated` (with `raw_translation` and the record's `engine`) or `blank_input`. Sleeps 1.5s per item when backend is `GOOGLE` (rate limit). The text arrives already rendered by Rust (`TranslationSource::render`, which owns the Chinese-script rule); post-processing happens in Rust after. No backend strips the terminator |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## No result cache

Translations are not cached. The utterance cache holds audio evidence only
(forced alignment, UTR ASR, Rev.AI transcripts and speaker evidence); text NLP
caching was removed because re-running warm inference cost less than the cache
lookups. Every translate run calls the worker.

---

## Worker IPC: translate task

```text
request item (rendered from that utterance's TranslationSource; the source and
target languages travel on the request envelope, not on each item):
{ "text": "bonjour le monde." }

TranslationResultV2 items, one per request item:
{ "kind": "translated", "raw_translation": "Hello world.", "engine": "googletrans-v1" }
{ "kind": "blank_input" }
{ "kind": "failed", "error": "Translation failed: ..." }
```

The PyO3 bridge parses each host item through the Rust wire type; an item that
does not parse becomes that item's `failed` outcome.

## What reaches the engine

`TranslationSource::render` is the only place source text is built, and three
closed rules decide its content, each a match with no catch-all:

| Rule | Sent | Not sent |
|---|---|---|
| Words (`TranslatableWordText::of_produced_word`) | Ordinary words and filled pauses (without the `&-` prefix); a replaced word contributes its replacement | `0`-prefixed and CA omissions, `&~` nonwords, `&+` fragments, `xxx` / `yyy` / `www` |
| Separators (`TranslatableSeparator::of_separator`) | The comma | The tag marker and vocative (`„`, `‡`), the CA prosodic marks |
| Terminator (`render_terminator`) | `.` (the ideographic full stop in Han script), `?` for every question-bearing variant including `+/?`, `+!?`, `+//?`, `+..?`, and `!` | `+...`, `+/.`, `+//.`, `+"/.`, `+".`, `+.` |

`RENDERED_TERMINATORS` lists every string the terminator rule can emit, and
`TranslationText::admit` refuses a translation equal to one of them, so an
engine echoing the punctuation batchalign3 sent cannot become a `%xtra` tier.

## Engine identity and provenance

The engine comes from the results, not from the worker's capability report.
`translate.rs` admits each item into `AdmittedTranslation`
(`Translated { text, engine }` or `BlankInput`), and the batch pipeline stamps
each file with `[fc-ba3 translate | engine=... ; lang=... | ...]`, naming the
distinct engines on the translations that file applied, joined with `+` in text
order (`result_named_provenance` with `ResultNamedCommand::Translate`, the
builder coref shares; it cannot fail). A file where nothing was translated
carries no stamp and the run says why (`TextStamp::NotStamped`). Because
nothing is read from the report, a translate job is never refused for a worker
that has not named its translation engine; that pre-dispatch refusal was
removed.

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

1. `common.engine_overrides.translate`: set by
   `--engine-overrides '{"translate":"<engine>"}'`.
2. `TranslateOptions.translate_engine: TranslateEngineName`: set by
   `--translate-engine google|tencent|aliyun|nllb|seamless`. Defaults
   to Google via `default_translate_engine()`.

There is deliberately no `server.yaml` knob for engine selection.
Translation engine is a policy choice, not a host fact, and policy
belongs at the invocation site (CLI flag or shell alias), never in
a config file. See the no-config-junk principle in
`book/src/batchalign/user-guide/commands/translate.md`.

The worker pool key includes the resolved translate engine
(`dispatch_engine_overrides_json` always emits a `translate` entry).
Google, Tencent, Aliyun, Seamless, and NLLB workers are not
interchangeable, so they end up in separate pools.

## Tencent backend specifics

The Tencent loader reuses the shared `read_asr_config()` helper at
`batchalign/inference/languages/cantonese/_common.py:77`, which
prefers `BATCHALIGN_TENCENT_{ID,KEY,REGION}` environment variables
(injected by the Rust control plane at worker spawn) and falls back
to `~/.batchalign.ini` `[asr]` section:

- `engine.tencent.id` → `TencentSecretId`
- `engine.tencent.key` → `TencentSecretKey`
- `engine.tencent.region` → `TencentRegion`

These are the same CAM credentials used by the Tencent ASR backend
(the `[asr]` section name is historical; the SecretId/SecretKey pair
authorizes any product the CAM user has permission for). The user
must have `tmt:TextTranslate` policy attached (e.g.,
`QcloudTMTFullAccess`), and the TMT product must be "opened" at the
Tencent Cloud account level, both are root-account / admin actions
on the Tencent side.

Rate-limit handling: the inference closure in
`batchalign/inference/translate.py` sleeps 0.2 s per item when the
backend is Tencent (5 QPS standard free-tier limit on
`TextTranslate`). This is the analogue of the existing 1.5 s
per-item sleep for Google.

Language-code handling: `_ISO_639_3_TO_TENCENT_LANG` (in
`batchalign/worker/_model_loading/translation.py`) maps the ISO-639-3
codes BA3 emits to Tencent's ISO-639-1 codes (`spa→es`, `cmn→zh`,
etc.). Unmapped source languages raise a clear `ValueError`
recommending `--translate-engine nllb`. Tencent does NOT list
`yue→en` in its supported pairs, Cantonese requests are rejected at
the table lookup, not at the API call.

Empty `SourceText` would be rejected by the Tencent API with a typed
`InvalidParameter` error. The loader short-circuits empty input
(returns the empty string) so a stray empty utterance doesn't surface
as a SDK exception that looks like a credentials problem.

## Aliyun backend specifics

The Aliyun loader (`_load_aliyun_translate`) uses the same shared
`read_asr_config()` helper, with credentials drawn from the
`BATCHALIGN_ALIYUN_AK_{ID,SECRET}` environment variables (injected
by the Rust control plane at worker spawn) or the
`~/.batchalign.ini` `[asr]` section:

- `engine.aliyun.ak_id` → Aliyun Access Key ID
- `engine.aliyun.ak_secret` → Aliyun Access Key Secret

These are the same access-key pair used by the Aliyun NLS ASR
backend. Aliyun MT does NOT need the `ak_appkey` field that NLS
ASR consumes, that key authorizes the WebSocket speech service,
not the REST translation service.

Region is pinned to `cn-hangzhou` (`_ALIYUN_MT_REGION` in
`translation.py`). Aliyun MT exposes a single global endpoint at
`mt.aliyuncs.com` across every supported region, so the AcsClient
region only affects request signing, there is no
`cn-hangzhou` vs `us-west-1` quality / availability split. If
region-pinning becomes a deployment concern later, promote to a
config-driven override.

SDK package: `aliyun-python-sdk-alimt` (pinned at `>=3.2.0` in
`pyproject.toml`). The loader uses the v20181012 General Translation
endpoint via `TranslateGeneralRequest` with `FormatType="text"` and
`Scene="general"` (both promoted to module-level constants
`_ALIYUN_MT_FORMAT_TYPE` / `_ALIYUN_MT_SCENE` so the wire shape is
visible without grepping for magic strings).

Language-code handling: `_ISO_639_3_TO_ALIYUN_LANG` (in
`batchalign/worker/_model_loading/translation.py`) maps the
ISO-639-3 codes BA3 emits to Aliyun's ISO-639-1-ish codes
(`spa→spa`, `cmn→zh`, `kor→ko`, **`yue→yue`**, etc.). The presence
of `yue` is the load-bearing reason this backend exists alongside
Tencent, see [User Guide: translate](../../user-guide/commands/translate.md)
for the operator-visible rationale.

Response envelope: Aliyun MT returns a JSON byte payload of the
shape `{"Code": "200", "Data": {"Translated": "...", "DetectedLanguage":
"...", "WordCount": "..."}, "RequestId": "..."}`. Non-`"200"` codes
surface as `ClientException`/`ServerException` from
`do_action_with_exception` before the loader parses; by the time
`json.loads` runs, `Code == "200"` is expected.

Empty `SourceText` short-circuits the same way Tencent does (return
empty string before any SDK call) for the same reason, Aliyun
treats empty input as an invalid request and would surface a typed
SDK exception that looks like a credentials problem.

End-to-end verification: the loader's SDK call shape is wired
against the `aliyun-python-sdk-alimt==3.2.0` source. Real-API smoke
testing happens at the operator boundary per the user-guide; CI
covers the wire shape via the mocked-SDK test in
`batchalign/tests/pipelines/translate/test_translation_model_loading.py::TestLoadAliyunTranslate`.

## BA2 → BA3 migration notes

| Concern | BA2-jan9 | BA3 |
|---------|----------|-----|
| CLI shape | `batchalign translate IN_DIR OUT_DIR` (separate dirs) | `batchalign3 translate <dir-or-file>` (in-place by default) |
| Default engine | `googletrans` (dispatch.py: `"translate": "gtrans"`) | `googletrans`, with explicit per-host opt-in to Seamless via `server.yaml` `default_translate_engine` or `--translate-engine seamless` |
| Concurrency | Sequential per utterance, with `time.sleep(1.5)` on Google | Batched cross-file dispatch, multiple worker groups per language, 1.5s sleep retained per-item on Google only |
| Re-run behavior | Skip already-translated utterances | Overwrite existing `%xtra` |
| What is sent | `utterance.strip(join_with_spaces=False, include_retrace=True, include_fp=True)` in `gtrans.py` and `seamless.py`: words, retraces, filled pauses and punctuation including the terminator, detokenized | The same words, as a typed `TranslationSource` rendered once at the wire boundary. Which words, which punctuation and how a terminator is written are BA3's own closed rules (see below). Before 2026-09-15 BA3 sent only `%mor`-domain words, with no retraces, no filled pauses and no terminator |
| Chinese preprocessing | Inline in `gtrans.py` only (spaces removed, `.` to `。`); `seamless.py` did NOT strip spaces (BA2 bug) | A property of the language: `WritingSystem::of_language` marks the Han-script varieties (`zho`, `cmn`, `yue`, `wuu`, `nan`, `hak`), and `TranslationSource::render` applies the rule for every backend |
| Empty translation | Dropped at injection: `generator.py` wrote `%xtra` only when the text was not `""`, `"."`, `"!"` or `"?"`, leaving the utterance with no tier | Refused when the result is admitted, as a typed per-item failure naming the engine and the remedy. Terminal, not retryable: an identical request gets an identical answer |
| Per-item failure | Aborts the file (single-file CLI invocation) | Marks the affected file as failed with a typed `TextWorkflowFileError::ItemErrors` carrying the engine error(s); other files in the same cross-file batch continue normally. Transient errors at the batch dispatch layer retry; per-item engine failures propagate to file-level failure without retry. |
| Output tier | `%xtra` | `%xtra` (identical) |

**Tier-name clarification.** Neither BA2 nor BA3 produces a `%tra` tier.
Both versions emit `%xtra`. Any other translation-tier name observed in
the wild was not written by Batchalign.

---

## Testing

```bash
make test
cargo test -p batchalign translate::
```

---

## Related developer documentation

- [Command Flowcharts: translate](../../architecture/command-flowcharts.md#translate)
