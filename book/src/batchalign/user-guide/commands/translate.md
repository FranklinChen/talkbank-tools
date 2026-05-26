# translate

**Status:** Current
**Last updated:** 2026-05-26 12:48 EDT

Add English translations to non-English CHAT transcripts by injecting a
`%xtra` tier after each utterance. Text-only — no audio involved.

## Engine

Four backends are available:

- **Google Translate** (`googletrans`) — calls the public Google Translate
  endpoint. Requires outbound reachability to `translate.google.com`;
  unsuitable for hosts behind the Great Firewall unless a VPN is active.
  Rate-limited to one item per 1.5 seconds inside the worker. **Default.**
- **Tencent Cloud TMT** (`tencent`) — China-friendly cloud API. Strong
  quality on Mandarin (`zh→en`); produces correct "Hello world" for
  `你好世界` where NLLB renders it "Good day". **Does NOT support
  Cantonese** (`yue→en`); route Cantonese through `nllb` instead.
  Requires CAM credentials with `tmt:TextTranslate` permission in
  `~/.batchalign.ini` `[asr]` section (`engine.tencent.id` / `key` /
  `region`), or via the `BATCHALIGN_TENCENT_{ID,KEY,REGION}`
  environment variables that the Rust control plane uses to inject
  fleet-managed credentials. Free tier: 5M characters/month;
  throttled to 5 QPS in-worker (0.2 s/item).
- **Meta NLLB-200-distilled-1.3B** (`nllb`) — runs locally in the Python
  worker. Model downloaded from HuggingFace on first use (~5 GB) and
  cached thereafter; no outbound network at inference time. **Best
  self-hosted fallback** — handles Cantonese first-class; for Mandarin
  short greetings prefer `tencent`. Long-form CJK is excellent. Runs
  unthrottled.
- **Meta SeamlessM4T** (`seamless`) — runs locally in the Python worker.
  BA2-inherited fallback. Empirical 2026-05-23 comparison found
  short-CJK quality is poor and the model hallucinates on empty inputs;
  **prefer `nllb` or `tencent` for new work.** Retained for back-compat.

Select with `--translate-engine google|tencent|nllb|seamless`. Default is
Google. Operators on hosts where Google Translate is unreachable pass
`--translate-engine tencent` (best Mandarin) or `--translate-engine nllb`
(handles Cantonese, no cloud account required) explicitly per invocation
(a shell alias is the right place to make that persistent for a given
user) — there is no per-host config file knob for engine selection, by
design.

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
# Translate a single file in place — source language is read from @Languages
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
    parse --> collect[collect_payloads\nExtract utterance text + source/target language]
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
| `--translate-engine google\|tencent\|nllb\|seamless` | `google` | Pick the translation engine for this invocation. `tencent` is best for Mandarin (requires CAM credentials, no Cantonese support); `nllb` is the recommended self-hosted fallback and handles Cantonese; `seamless` is BA2-inherited and retained for back-compat. |
| `--merge-abbrev` / `--no-merge-abbrev` | off | Merge abbreviations in the translated output |

---

## Failure modes

batchalign3 translate fails fast on engine failures rather than emitting
partial output. When the worker reports a per-utterance error (engine
network failure, GFW block on Google, rate-limit exhaustion, model
runtime error), the affected file is marked failed with a typed
`ItemErrors` message naming the first few offending items and the
total count. Other files in the same batch continue normally — one
bad file does not poison the rest (BA2-parity multi-file semantics).

The output `.cha` for a failed file is **not** written. There is no
silent path where a job appears successful but produced a `.cha`
with missing `%xtra` tiers — if a tier is missing, the job result
will say so.

### Common cases

| Situation | What happens |
| --- | --- |
| Google Translate unreachable (GFW block, network outage, DNS failure) | File marked failed with `translate failed for N item(s): item 0: Translation failed: ConnectionResetError ...`. Use `--translate-engine tencent` (best Mandarin quality, requires CAM credentials) or `--translate-engine nllb` (self-hosted, handles Cantonese). |
| Rate-limit (429) on one or more items | File marked failed citing the 429 message verbatim. Retry; if persistent, switch to `--translate-engine tencent` or `--translate-engine nllb` or split the workload. |
| Self-hosted model first-download (HuggingFace) fails | File marked failed with the underlying HF error. If on a host where the default HF endpoint is slow, set `HF_ENDPOINT=https://hf-mirror.com` before the worker starts. Applies to both `nllb` (~5 GB) and `seamless` (~1.2 GB). |
| Tencent CAM credentials missing / wrong | File marked failed citing `~/.batchalign.ini` parse error or `AuthFailure.UnauthorizedOperation`. Ensure `engine.tencent.id`/`key`/`region` are populated and the CAM user has `tmt:TextTranslate` policy attached. The TMT product itself must also be "opened" at the Tencent Cloud account level (`FailedOperation.UserNotRegistered` indicates this is missing). |
| Tencent `yue→en` request | Raises `ValueError: Tencent TMT does not support source language 'yue'; use --translate-engine nllb`. Switch the Cantonese run to `nllb`. |
| googletrans library import error in a stripped venv | Worker startup fails (loud), not a per-job failure. |

---

## What changes in the `.cha` file

- A `%xtra:` tier is added after each utterance containing the English
  translation
- All other tiers (`%mor`, `%gra`, `%wor`) are preserved unchanged
- No audio is involved

---

## Related documentation

- [Command I/O: translate](../../reference/command-io.md#6-translate) — I/O patterns and mutation behavior
- [Command Flowcharts: translate](../../architecture/command-flowcharts.md#translate) — full architecture flowchart
