# utseg

**Status:** Current
**Last updated:** 2026-04-25 14:51 EDT

Re-segment utterance boundaries in an existing CHAT transcript. Text-only
— no audio involved. The model selected per language is either a trained
BERT per-word boundary classifier (eng / zho / yue) or, for other
languages, Stanza constituency parsing where it is available.

`transcribe` already runs this same step at the end of every run
(`with_utseg = true` is the default in the transcribe pipeline). The
standalone `utseg` command is for already-existing corpora — files
transcribed elsewhere, hand-typed transcripts, or older BA2 output —
where utterances run on into long blobs and need to be split.

---

## Quick start

```bash
# Re-segment a single file in place
batchalign3 utseg file.cha --lang eng

# Re-segment a corpus directory
batchalign3 utseg corpus/ -o segmented/ --lang eng

# Use the fleet server
batchalign3 --server http://net:8001 utseg corpus/ -o out/ --lang eng
```

---

## Pipeline

All utterances across all input files are pooled into a single GPU batch.

```mermaid
flowchart TD
    start([utseg invoked]) --> parse[Parse all files → ASTs]
    parse --> collect[collect_payloads\nExtract word sequences per utterance]
    collect --> worker[execute_v2(task="utseg")\nprepared_text batch → BERT assignments\nor Stanza constituency trees]
    worker --> apply[Apply segmentation\nSplit/merge utterances at predicted boundaries]
    apply --> merge_check{--merge-abbrev?}
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

If you combine `--file-list` with in-place processing on a large corpus, do
not expect the `.cha` files on disk to rewrite one by one during the run.
`utseg` batches and stages text-NLP work internally; the visible in-place file
updates may land only when the current invocation finishes. For long reruns
where you want output to appear incrementally, split the file list into
smaller chunks and run those chunks sequentially.

### utseg options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | `eng` | 3-letter ISO language code |
| `-n`, `--num-speakers N` | `2` | Number of speakers |
| `--merge-abbrev` | off | Merge abbreviations in the output |

---

## What changes in the `.cha` file

- Utterance boundaries (`*SPK:` lines) are recomputed — utterances may be
  split or merged
- Existing `%mor` and `%gra` tiers on recomputed utterances will be
  invalidated; re-run `morphotag` after `utseg` if those tiers are needed
- No audio is involved

---

## Language support

Per-language model selection is driven by `_RESOLVER["utterance"]` in
`batchalign/models/resolve.py`:

| `--lang` | Model loaded | Source |
|----------|--------------|--------|
| `eng` | `talkbank/CHATUtterance-en` (BERT per-word classifier) | TalkBank fine-tune |
| `zho` (Mandarin) | `talkbank/CHATUtterance-zh_CN` (BERT) | TalkBank fine-tune |
| `yue` (Cantonese) | `PolyU-AngelChanLab/Cantonese-Utterance-Segmentation` (BERT) | PolyU AngelChanLab |
| any other language | Stanza constituency parser, where available | Stanza |

The English BERT is **not** applied cross-lingually — running `utseg
--lang fra` does not load `CHATUtterance-en`. Languages with no entry in
the resolver fall through to the Stanza constituency path. Stanza ships
constituency models for ~11 languages (en, de, es, it, pt, da, id, ja,
tr, vi, zh-hans); for any other language, `utseg` currently produces
no splits and the file passes through unchanged.

See [Utterance Segmentation](../../reference/utterance-segmentation.md)
for the algorithm details and the
[Stanza Capability Registry](../../architecture/stanza-capability-registry.md)
for the per-language processor availability table.

---

## Related documentation

- [Utterance Segmentation](../../reference/utterance-segmentation.md) — algorithm and model details
- [Stanza Capability Registry](../../architecture/stanza-capability-registry.md) — which languages support constituency parsing
- [Command I/O: utseg](../../reference/command-io.md#5-utseg) — I/O patterns and mutation behavior
- [Command Flowcharts: utseg](../../architecture/command-flowcharts.md#utseg) — full architecture flowchart
