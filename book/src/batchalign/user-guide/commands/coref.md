# coref

**Status:** Current
**Last updated:** 2026-05-23 09:20 EDT

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

# Use the remote server
batchalign3 --server http://your-server:8001 coref corpus/ -o out/
```

---

## Pipeline

`coref` does not use the utterance cache. Note that no text-NLP command
caches either (`CacheTaskName` at
`crates/batchalign/src/chat_ops/cache_key.rs:58` covers only
`ForcedAlignment` and `UtrAsr`), so this is consistent with
`morphotag`/`utseg`/`translate`. What's specific to `coref` is the
*reason*: coreference chains span the entire document, so a
per-utterance cache key would be unsound even if the infrastructure
existed — the same utterance has different coreference in different
document contexts.

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
| `--merge-abbrev` / `--no-merge-abbrev` | off | Merge abbreviations in the output |

Source language is read from each file's `@Languages` header.
Non-English files pass through unchanged (Stanza's coreference model
is English-only).

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

**No caching.** Re-running `coref` always calls the worker. This is
true of every text-NLP command — `morphotag`, `utseg`, and `translate`
also re-run from scratch each time — so this is not a coref-specific
slowdown vs the others. What is specific to coref is the
document-level scope: even if a per-task text-NLP cache were added
later, coref's cache key would have to include the entire document
because coreference depends on full context.

**Best suited for local or direct-server execution.** `coref` is a
document-level workflow that benefits from locality. It is not an interactive
remote-server command in the same way as `align` or `transcribe`.

---

## Failure modes

coref fails fast on engine failures rather than emitting silent
no-coref output. When the Stanza coref worker reports a per-file
error (model runtime error, protocol violation, batch IPC failure),
the affected file is marked failed with a typed `ItemErrors` message
carrying the engine error verbatim. A batch-level coref failure
(worker spawn / IPC) marks every English-eligible file in the same
batch as failed; non-eligible files (dummy or non-English) pass
through unchanged. The output `.cha` for a failed file is **not**
written — there is no path where the file appears successful but
the `%xcoref` tier is silently missing.

---

## Related documentation

- [Command I/O: coref](../../reference/command-io.md#7-coref) — I/O patterns and mutation behavior
- [Command Flowcharts: coref](../../architecture/command-flowcharts.md#coref) — full architecture flowchart
