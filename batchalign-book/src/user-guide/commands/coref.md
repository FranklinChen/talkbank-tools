# coref

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Add sparse coreference annotation tiers (`%xcoref`) to CHAT transcripts.
English-only. Uses full document context — all utterances in the file are
processed together as a single document. Text-only — no audio involved.

---

## Quick start

```bash
# Annotate a single file in place
batchalign3 coref file.cha

# Annotate a corpus directory
batchalign3 coref corpus/ -o coref-output/

# Use the fleet server
batchalign3 --server http://net:8001 coref corpus/ -o out/
```

---

## Pipeline

Unlike other commands, `coref` does **not** use the utterance cache.
Coreference chains span the entire document, making per-utterance cache keys
meaningless — the same utterance has different coreference in different document
contexts.

```mermaid
flowchart TD
    start([coref invoked]) --> parse[Parse all files → ASTs]
    parse --> collect[collect_payloads\nExtract sentences — full document context]
    collect --> worker[execute_v2(task="coref")\nprepared_text batch → structured chain refs]
    worker --> inject[inject %xcoref tiers — sparse\nOnly utterances with coreferent mentions]
    inject --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| serialize
    merge --> serialize[Serialize → .cha output]
    serialize --> done([Output .cha files])

    style collect fill:#ffd,stroke:#aa0
    note1[No caching — full-document context\nmakes per-utterance keys meaningless]
    collect --- note1
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

### coref options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | from `@Languages` | 3-letter ISO language code. Overrides `@Languages` when set |
| `--merge-abbrev` | off | Merge abbreviations in the output |

---

## What changes in the `.cha` file

- `%xcoref:` tiers are added sparsely — only on utterances that contain
  mentions participating in a coreference chain
- All other tiers are preserved unchanged
- No audio is involved

---

## Gotchas

**English-only.** Non-English files pass through without modification.
Stanza's coreference model is only available for English.

**No caching.** Because coreference is a document-level operation, results
cannot be cached per utterance. Re-running `coref` always calls the worker.
For large corpora this means longer processing times compared to `morphotag`.

**Best suited for local or direct-server execution.** `coref` is a
document-level workflow that benefits from locality. It is not an interactive
remote-server command in the same way as `align` or `transcribe`.

---

## Related documentation

- [Command I/O: coref](../../reference/command-io.md#7-coref) — I/O patterns and mutation behavior
- [Command Flowcharts: coref](../../architecture/command-flowcharts.md#coref) — full architecture flowchart
