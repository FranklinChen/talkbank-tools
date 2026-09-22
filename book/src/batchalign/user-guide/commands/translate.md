# translate

**Status:** Current
**Last updated:** 2026-09-15 20:20 EDT

Add English translations to non-English CHAT transcripts by injecting a
`%xtra` tier after each utterance. Text-only, no audio involved.

## What gets translated

What was spoken. Each utterance is sent as every word the speaker produced, in
order, followed by the utterance's terminator, so a question is translated as a
question. Retraced words (`<I like> [/]`) and filled pauses (`&-um`, sent as
`um`) are included, which is what batchalign2 did; a word the transcriber
replaced (`hafta [: have to]`) is sent as the replacement.

Left out, because they are not words a translator can read: `0`-prefixed
omissions (recorded as not said), `&~` nonwords, `&+` fragments, and the
untranscribed markers `xxx` / `yyy` / `www`. An utterance that produced no words
at all is not sent and gets no `%xtra` tier.

Only punctuation an ordinary reader would recognise travels with the words: the
comma, and a terminator that is a period, question mark or exclamation mark
(the question-bearing CHAT terminators `+/?`, `+!?`, `+//?` and `+..?` are sent
as a question mark). CHAT-only notation is not sent at all, so `+...`, `+/.`,
`+//.`, the tag marker and the vocative never reach the engine as text to
translate.

For languages written in Han script (`zho`, `cmn`, `yue`, `wuu`, `nan`, `hak`)
the words are joined without spaces and the period is written as the ideographic
full stop; the translation is converted back on the way in. This follows the
writing system, not a list of two codes, so Mandarin tagged `cmn` is handled
like `zho`.

Before 2026-09-15 batchalign3 sent only the `%mor`-domain words: no retraces, no
filled pauses and no terminator. Files translated by an earlier build were
translated from a different source text; re-run `translate` over them to get
translations of what was actually said.

## Engine

Five backends are available:

- **Google Translate** (`googletrans`), calls the public Google Translate
  endpoint. Requires outbound reachability to `translate.google.com`;
  unsuitable for hosts behind the Great Firewall unless a VPN is active.
  Rate-limited to one item per 1.5 seconds inside the worker. **Default.**
- **Tencent Cloud TMT** (`tencent`), China-friendly cloud API. Strong
  quality on Mandarin (`zh→en`); produces correct "Hello world" for
  `你好世界` where NLLB renders it "Good day". **Does NOT support
  Cantonese** (`yue→en`); route Cantonese through `aliyun` (cloud) or
  `nllb` (self-hosted) instead.
  Requires CAM credentials with `tmt:TextTranslate` permission in
  `~/.batchalign.ini` `[asr]` section (`engine.tencent.id` / `key` /
  `region`), or via the `BATCHALIGN_TENCENT_{ID,KEY,REGION}`
  environment variables that the Rust control plane uses to inject
  fleet-managed credentials. Free tier: 5M characters/month;
  throttled to 5 QPS in-worker (0.2 s/item).
- **Aliyun Machine Translation** (`aliyun`), China-friendly cloud API
  via Alibaba Cloud's `alimt` General service. **Supports Cantonese
  (`yue→en`)**: the canonical cloud option for HK material, where
  Tencent TMT does not list Cantonese as a source language. Requires
  Aliyun access-key credentials in `~/.batchalign.ini` `[asr]` section
  (`engine.aliyun.ak_id` / `ak_secret`, shared with the Aliyun ASR
  backend), or via the `BATCHALIGN_ALIYUN_AK_{ID,SECRET}` environment
  variables. Region is pinned to `cn-hangzhou` (Aliyun MT exposes one
  global endpoint at `mt.aliyuncs.com`, so the region only affects
  request signing). Quotas and pricing per Aliyun MT service terms.
- **Meta NLLB-200-distilled-1.3B** (`nllb`), runs locally in the Python
  worker. Model downloaded from HuggingFace on first use (~5.5 GB) and
  cached thereafter; no outbound network at inference time. **Best
  self-hosted fallback**: handles Cantonese first-class; for Mandarin
  short greetings prefer `tencent`. Long-form CJK is excellent. Runs
  unthrottled.
- **Meta SeamlessM4T** (`seamless`), runs locally in the Python worker.
  BA2-inherited fallback. Empirical 2026-05-23 comparison found
  short-CJK quality is poor and the model hallucinates on empty inputs;
  **prefer `nllb` or `tencent` for new work.** Retained for back-compat.

Select with `--translate-engine`; `batchalign3 translate --help` lists every
accepted value, derived from the engine enum, and rejects anything else while
parsing.
Default is Google. Operators on hosts where Google Translate is
unreachable pass `--translate-engine tencent` (best Mandarin),
`--translate-engine aliyun` (Cantonese-capable cloud option), or
`--translate-engine nllb` (self-hosted, handles Cantonese, no cloud
account required) explicitly per invocation (a shell alias is the right
place to make that persistent for a given user), there is no per-host
config file knob for engine selection, by design.

For symmetry with how ASR and FA engines are selected, the shared
`--engine-overrides '{"translate":"<engine>"}'` global flag also
works and takes precedence over `--translate-engine`.

### Migrating from BA2

BA2 read the translation engine from `~/.batchalign.ini`:

```ini
[translate]
engine = seamless_translate
```

BA3 does not honor that key. The replacement is the explicit CLI flag
`--translate-engine seamless` (or the shared
`--engine-overrides '{"translate":"seamless"}'`). If you previously
relied on the INI entry for routine runs, drop the line from
`~/.batchalign.ini` and add the flag to whatever wrapper or alias you
invoke `batchalign3 translate` through.

## Re-running on already-translated files

Running `translate` on a file that already has `%xtra` tiers will
**overwrite** them with fresh output. This is a deliberate change from
batchalign2, which preserved the first translation and skipped any
utterance that already had one. If you want to keep prior translations,
copy the file first or filter your inputs.

---

## Quick start

```bash
# Translate a single file in place, source language is read from @Languages
batchalign3 translate file.cha

# Translate a corpus directory
batchalign3 translate corpus/ -o translated/

# Use the remote server
batchalign3 --server http://your-server:8001 translate corpus/ -o out/
```

`translate` has **no `--lang` flag**. Source language for each file is
read from that file's own `@Languages:` header. Translation target is
fixed to English. To "override" the source language, edit the file's
`@Languages:` line.

---

## Pipeline

```mermaid
flowchart TD
    start([translate invoked]) --> parse[Parse all files → ASTs]
    parse --> collect[collect_payloads\nSpoken words and terminator per utterance]
    collect --> worker[execute_v2(task="translate")\nprepared_text batch → raw translations]
    worker --> inject[inject %xtra tiers with translated text]
    inject --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| serialize
    merge --> serialize[Serialize → .cha output]
    serialize --> done([Output .cha files])
```

Translation results are not cached: the `CacheTaskName` enum (at
`crates/batchalign/src/chat_ops/cache_key.rs:58`) only has
`ForcedAlignment` and `UtrAsr` variants, and `translate.rs` does not
call `cache.put`. Repeated `translate` runs on the same input
re-invoke the worker.

---

## Options

### Path options

| Option | Meaning |
| --- | --- |
| `PATHS...` | Input `.cha` files or directories |
| `-o`, `--output DIR` | Output directory (omit to overwrite in place) |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Explicit in-place flag |

### translate options

| Option | Default | Meaning |
| --- | --- | --- |
| `--translate-engine google\|tencent\|aliyun\|nllb\|seamless` | `google` | Pick the translation engine for this invocation. `tencent` is best for Mandarin (requires CAM credentials, no Cantonese support); `aliyun` is the Cantonese-capable cloud option (requires Aliyun access keys); `nllb` is the recommended self-hosted fallback and handles Cantonese; `seamless` is BA2-inherited and retained for back-compat. |
| `--merge-abbrev` / `--no-merge-abbrev` | off | Merge abbreviations in the translated output |

---

## Provenance

Every file translate writes records the engines that produced its translations
in a `[fc-ba3 translate | engine=... ; lang=... | ...]` comment. Each engine is
the one the worker named on the translation it returned, so a file where
nothing was translated (for example, only blank utterances) gets no comment and
the run says why. Files translated by a build before 2026-09-15 carry no
comment at all, because the batch path wrote none; re-running `translate` over
them adds one. See [Processing Provenance](../provenance.md).

---

## Failure modes

batchalign3 translate fails fast on engine failures rather than emitting
partial output. When the worker reports a per-utterance error (engine
network failure, GFW block on Google, rate-limit exhaustion, model
runtime error), the affected file is marked failed with a typed
`ItemErrors` message naming the first few offending items and the
total count. Other files in the same batch continue normally, one
bad file does not poison the rest (BA2-parity multi-file semantics).

The output `.cha` for a failed file is **not** written. There is no
silent path where a job appears successful but produced a `.cha`
with missing `%xtra` tiers, if a tier is missing, the job result
will say so.

### Common cases

| Situation | What happens |
| --- | --- |
| Google Translate unreachable (GFW block, network outage, DNS failure) | File marked failed with `translate failed for N item(s): item 0: Translation failed: ConnectionResetError ...`. Use `--translate-engine tencent` (best Mandarin quality, requires CAM credentials) or `--translate-engine nllb` (self-hosted, handles Cantonese). |
| Rate-limit (429) on one or more items | File marked failed citing the 429 message verbatim. Retry; if persistent, switch to `--translate-engine tencent` or `--translate-engine nllb` or split the workload. |
| Self-hosted model first-download (HuggingFace) fails | File marked failed with the underlying HF error. If on a host where the default HF endpoint is slow, set `HF_ENDPOINT=https://hf-mirror.com` before the worker starts. Applies to both `nllb` (~5.5 GB) and `seamless` (~4.8 GB). |
| Tencent CAM credentials missing / wrong | File marked failed citing `~/.batchalign.ini` parse error or `AuthFailure.UnauthorizedOperation`. Ensure `engine.tencent.id`/`key`/`region` are populated and the CAM user has `tmt:TextTranslate` policy attached. The TMT product itself must also be "opened" at the Tencent Cloud account level (`FailedOperation.UserNotRegistered` indicates this is missing). |
| Tencent `yue→en` request | Raises `ValueError: Tencent TMT does not support source language 'yue'; use --translate-engine aliyun (cloud, supports Cantonese) or --translate-engine nllb (self-hosted local model)`. Switch the Cantonese run to `aliyun` or `nllb`. |
| Aliyun MT credentials missing / wrong | File marked failed citing `~/.batchalign.ini` parse error or an Aliyun SDK `ClientException`/`ServerException`. Ensure `engine.aliyun.ak_id` / `ak_secret` are populated (same keys the Aliyun ASR backend uses); the Aliyun MT service must also be activated in the Alibaba Cloud console for the access key's account. |
| Aliyun MT unmapped source language | Raises `ValueError: Aliyun MT does not have a mapped source language for '<iso>'; use --translate-engine nllb for this language`. Use `nllb` for the unmapped language or extend `_ISO_639_3_TO_ALIYUN_LANG` in `batchalign/worker/_model_loading/translation.py`. |
| Engine returns an empty translation for an utterance (nothing, or only the punctuation batchalign3 sent) | File marked failed with `translate failed for N item(s): item 0: <engine> returned an empty translation; try a different --translate-engine or options and run the file again`. The file is not written. The verdict is terminal, because the same request gets the same answer from the same engine: the remedy is another engine or different options, not a retry. Before 2026-09-15 the utterance was silently left without a `%xtra` tier. |
| googletrans library import error in a stripped venv | Worker startup fails (loud), not a per-job failure. |

---

## What changes in the `.cha` file

- A `%xtra:` tier is added after each utterance that produced words, containing
  the English translation
- An utterance the engine returned nothing usable for does not get an empty
  tier: the file fails instead, and nothing is written
- All other tiers (`%mor`, `%gra`, `%wor`) are preserved unchanged
- No audio is involved

---

## Related documentation

- [Command I/O: translate](../../reference/command-io.md#6-translate), I/O patterns and mutation behavior
- [Command Flowcharts: translate](../../architecture/command-flowcharts.md#translate), full architecture flowchart
