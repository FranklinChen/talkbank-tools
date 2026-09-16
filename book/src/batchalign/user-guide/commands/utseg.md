# utseg

**Status:** Current
**Last updated:** 2026-09-16 08:18 EDT

Re-segment utterance boundaries in an existing CHAT transcript. Text-only
, no audio involved. The model selected per language is either a trained
BERT per-word boundary classifier (eng / cmn,zho / yue) or, for other
languages, Stanza constituency parsing where it is available.

`transcribe` already runs this same step at the end of every run
(`with_utseg = true` is the default in the transcribe pipeline). The
standalone `utseg` command is for already-existing corpora, files
transcribed elsewhere, hand-typed transcripts, or older BA2 output,
where utterances run on into long blobs and need to be split.

---

## Quick start

```bash
# Re-segment a single file in place
batchalign3 utseg file.cha --lang eng

# Re-segment a corpus directory
batchalign3 utseg corpus/ -o segmented/ --lang eng

# Use the remote server
batchalign3 --server http://your-server:8001 utseg corpus/ -o out/ --lang eng
```

---

## Pipeline

Each file is dispatched on its own, `dispatch_utseg_job` in
`crates/batchalign/src/execution/utseg.rs` calls
`gateway.utseg_batch(&[one_file], lang)` per file and writes that
file's result to disk before starting the next. (This replaced an
earlier "pool everything across all files, batch through one worker,
write at end" pattern, which lost the entire run's work on a daemon
redeploy mid-batch. The per-file shape limits a mid-run interruption
to losing only files currently in flight.) Per-file concurrency is
bounded by `plan.kernel_plan.file_parallelism_hint` (clamped to ≥ 1),
the same heuristic as `fa_pipeline.rs`.

```mermaid
flowchart TD
    start([utseg invoked]) --> parse[Parse one file → AST]
    parse --> collect[collect_payloads\nExtract word sequences per utterance]
    collect --> worker[gateway.utseg_batch(&[file], lang)\n→ BERT assignments\nor Stanza constituency trees]
    worker --> apply[Apply segmentation\nSplit/merge utterances at predicted boundaries]
    apply --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| serialize
    merge --> serialize[Serialize → .cha output]
    serialize --> done([Write file's .cha; next file in pool])
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

In-place rewrites with `--file-list` on a large corpus do appear
file-by-file as the run progresses (each file is written to disk
before the next file's worker call starts). This is a deliberate
property of the per-file dispatch shape, interruption mid-run loses
only the files currently in flight, not the entire batch. Splitting
the file list into smaller chunks is therefore unnecessary for
incremental visibility, though it remains useful for managing memory
or scheduling.

### utseg options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | `eng` | 3-letter ISO language code |
| `--num-speakers N` | `2` | Number of speakers. No short flag; `-n` was removed 2026-08-19. |
| `--merge-abbrev` | off | Merge abbreviations in the output |

---

## What changes in the `.cha` file

- Utterance boundaries (`*SPK:` lines) are recomputed, utterances may be
  split or merged
- Existing `%mor` and `%gra` tiers on recomputed utterances will be
  invalidated; re-run `morphotag` after `utseg` if those tiers are needed
- No audio is involved

The boundary model uses lexical context only. It does not receive audio pause,
energy, pitch, or diarization evidence. Internally BA3 validates one assignment
per input word and retains typed model evidence across the worker boundary.
The versioned `--debug-dir` evidence sidecars described in the transcribe guide
belong to transcribe's distinct pre-CHAT and post-CHAT phases; standalone
`utseg` does not currently write those sidecars.

---

## Language support

Per-language model selection is driven by `UTSEG_BOUNDARY_MODELS` in
`crates/batchalign/src/model_manifest.rs`, which also pins the exact revision
each one loads:

| `--lang` | Model loaded | Source |
|----------|--------------|--------|
| `eng` | `talkbank/CHATUtterance-en` (BERT per-word classifier) | TalkBank fine-tune |
| `cmn` / `zho` (Mandarin) | `talkbank/CHATUtterance-zh_CN` (BERT) | TalkBank fine-tune |
| `yue` (Cantonese) | `PolyU-AngelChanLab/Cantonese-Utterance-Segmentation` (BERT) | PolyU AngelChanLab |
| any other language | refused by default; opt in via `--utseg-fallback-stanza` | Stanza |

The English BERT is **not** applied cross-lingually, running `utseg
--lang fra` does not load `CHATUtterance-en`. For any language without
a TalkBank BERT model in the table above, `utseg` refuses the
substitution by default, and **the job is refused when it is planned**:
the language and the fallback policy are both known before any work is
dispatched, so the run stops there with a message naming the language.
Nothing is written, and the input is not quietly copied through as if it
had been segmented. To permit the legacy Stanza constituency-parser
fallback (the same segmenter Batchalign 2 used for unsupported
languages), pass `--utseg-fallback-stanza`:

```bash
batchalign3 utseg corpus-fra/ --lang fra --utseg-fallback-stanza
```

Quality varies, Stanza ships constituency models for ~11 languages
(en, de, es, it, pt, da, id, ja, tr, vi, zh-hans). The opt-in design
prevents accidental quality regressions on unsupported languages.

See [Utterance Segmentation](../../reference/utterance-segmentation.md)
for the algorithm details and the
[Stanza Capability Registry](../../architecture/stanza-capability-registry.md)
for the per-language processor availability table.

---

## Provenance

Every file utseg writes records what segmented it in a
`[fc-ba3 utseg | engine=... ; lang=... | ...]` comment. `engine=` is one of:

| Value | Meaning |
| --- | --- |
| `<model id>@<revision>` | The TalkBank boundary model, with the exact revision the worker loaded (for example `talkbank/CHATUtterance-en@764ec3f...`) |
| `stanza-constituency` | The opt-in Stanza constituency fallback (`--utseg-fallback-stanza`) |

A boundary model is always written with its revision. It is loaded from a
pinned snapshot and the commit is read off the directory on disk, so a worker
that cannot say which revision it loaded refuses the job rather than reporting
a bare id. Files written before that pin landed may carry an id with no
revision; re-running `utseg` over them records the full identity.

Several sources on one file are joined with `+` in text order. There is no
placeholder: a worker that returns boundaries without naming what produced them
leaves the file with no comment at all, and the run says so.

Files segmented by a build before 2026-09-15 carry no comment, because the
standalone command wrote none; re-running `utseg` over them adds one. See
[Processing Provenance](../provenance.md).

---

## Failure modes

utseg fails fast on engine failures rather than emitting partial
output. When the BERT or Stanza worker reports a per-utterance error
(model runtime error, malformed constituency tree, protocol
violation), the affected file is marked failed with a typed
`ItemErrors` message naming the first few offending items and the
total count. Per-file dispatch (`utseg`-specific, BA3 utseg
deliberately does NOT cross-file-batch) means one failing file has
no effect on the next file in a multi-file run. The output `.cha`
for a failed file is **not** written.

---

## Related documentation

- [Utterance Segmentation](../../reference/utterance-segmentation.md), algorithm and model details
- [Stanza Capability Registry](../../architecture/stanza-capability-registry.md), which languages support constituency parsing
- [Command I/O: utseg](../../reference/command-io.md#5-utseg), I/O patterns and mutation behavior
- [Command Flowcharts: utseg](../../architecture/command-flowcharts.md#utseg), full architecture flowchart
