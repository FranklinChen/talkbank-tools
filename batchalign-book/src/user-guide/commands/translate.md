# translate

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Add English translations to non-English CHAT transcripts by injecting a
`%xtra` tier after each utterance. Text-only — no audio involved.

---

## Quick start

```bash
# Translate a single file in place
batchalign3 translate file.cha

# Translate a corpus directory
batchalign3 translate corpus/ -o translated/

# Override the source language (file has wrong or missing @Languages header)
batchalign3 translate corpus/ -o out/ --lang spa

# Use the fleet server
batchalign3 --server http://net:8001 translate corpus/ -o out/
```

---

## Pipeline

```mermaid
flowchart TD
    start([translate invoked]) --> parse[Parse all files → ASTs]
    parse --> collect[collect_payloads\nExtract utterance text + source/target language]
    collect --> cache[Cache lookup — BLAKE3 keys\ntext + src_lang + tgt_lang]
    cache --> worker[execute_v2(task="translate") misses\nprepared_text batch → raw translations]
    worker --> inject[inject %xtra tiers with translated text]
    inject --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| serialize
    merge --> serialize[Serialize → .cha output]
    serialize --> done([Output .cha files])
```

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
| `--lang CODE` | from `@Languages` | 3-letter ISO source language code. Overrides the file's `@Languages` header when set |
| `--merge-abbrev` | off | Merge abbreviations in the output |

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
