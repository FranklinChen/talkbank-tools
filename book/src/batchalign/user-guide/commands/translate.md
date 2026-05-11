# translate

**Status:** Current
**Last updated:** 2026-05-11 10:09 EDT

Add English translations to non-English CHAT transcripts by injecting a
`%xtra` tier after each utterance. Text-only — no audio involved.

## Engine

The worker picks Google Translate (`googletrans`) at startup if the library
is importable, and silently falls back to Meta SeamlessM4T
(`facebook/hf-seamless-m4t-medium`) if `googletrans` cannot be imported.
There is no user-facing flag to choose — the selection is determined at
worker boot. Google calls are rate-limited to one item per 1.5 seconds
inside the worker; Seamless runs unthrottled.

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
| `--merge-abbrev` / `--no-merge-abbrev` | off | Merge abbreviations in the translated output |

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
