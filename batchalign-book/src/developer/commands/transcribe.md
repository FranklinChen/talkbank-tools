# transcribe — Developer Reference

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Implementation guide for the `transcribe` command. For user-facing
documentation, see [User Guide: transcribe](../../user-guide/commands/transcribe.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign-cli/src/args/commands.rs` — `TranscribeArgs` | ASR engine, diarization, lang, num-speakers |
| Options builder | `crates/batchalign-cli/src/args/options.rs` — `build_transcribe_options()` | Maps `TranscribeArgs` → `CommandOptions::Transcribe(TranscribeOptions)` |
| Command definition | `crates/batchalign-app/src/commands/transcribe.rs` | `CommandDefinition` impl |
| Transcribe pipeline | `crates/batchalign-app/src/runner/dispatch/transcribe_pipeline.rs` | Per-file orchestration: ASR → post-processing → utseg → optional morphotag |
| ASR post-processing | `crates/batchalign-chat-ops/src/asr_postprocess/` | Compound merging, number expansion, Cantonese normalization, retokenization |
| CHAT assembly | `crates/batchalign-chat-ops/src/transcribe.rs` | `build_chat()` — assembles `ChatFile` from raw ASR tokens |
| Speaker reassignment | `crates/batchalign-chat-ops/src/transcribe.rs` — `reassign_speakers()` | Rewrites speaker codes + headers from diarization segments |
| ASR worker IPC | `batchalign/inference/asr.py` | Whisper/Rev.AI ASR, returns raw tokens |
| Speaker worker IPC | `batchalign/inference/speaker.py` — `batch_infer_speaker()` | Pyannote/NeMo diarization, returns speaker segments |

---

## ASR post-processing chain

All post-processing runs in Rust (`crates/batchalign-chat-ops/src/asr_postprocess/`):

```
1. Compound merging           — rejoin compound words split by ASR
2. Timed word extraction      — seconds → milliseconds
3. Multi-word token splitting — interpolate timestamps for split MWT
4. Number expansion           — digits → word form (single Rust per-word pass)
4b. Cantonese normalization   — simplified→HK traditional + domain replacements (yue only)
5. Long-turn splitting        — chunk at >300 words
6. Retokenization             — punctuation-based utterance splitting
```

Number expansion (step 4) runs entirely in Rust. Cardinals route through
the per-language `NUM2LANG` registry (47 languages); English ordinals
and decades use the deterministic `ordinal_year_eng` composer; CJK
languages use `num2chinese`; currency, percent, dash-ranges, and
digit-leading hyphens have dedicated Rust handlers. The Python
`num2words` IPC was removed 2026-04-26.

Cantonese normalization (step 4b) uses the `zhconv` crate and an Aho-Corasick
replacement table. Implementation: `asr_postprocess/cantonese.rs`.

---

## Worker IPC: ASR task (V2 protocol)

```
execute_v2 request:
{
  "task": "asr",
  "prepared_audio": { path, start_ms, end_ms, sample_rate },
  "engine": "rev" | "whisper" | "whisperx" | "whisper_oai" | "tencent" | ...,
  "language": "eng",
  "num_speakers": 2
}

execute_v2 response:
{
  "tokens": [
    { "word": "hello", "start_s": 0.12, "end_s": 0.45,
      "speaker": "SPEAKER_00", "confidence": 0.98 },
    ...
  ]
}
```

The speaker field is optional — Rev.AI always provides it; Whisper omits it.

## Worker IPC: speaker task (V2 protocol)

When `--diarization enabled` is set, a second worker call runs after ASR:

```
execute_v2 request:
{
  "task": "speaker",
  "prepared_audio": { path, ... },
  "num_speakers": 2
}

execute_v2 response:
{
  "segments": [
    { "start_s": 0.0, "end_s": 2.3, "speaker": "SPEAKER_00" },
    ...
  ]
}
```

`reassign_speakers()` in `transcribe.rs` then relabels utterances using
these segments as the authoritative source.

---

## Rev.AI `skip_postprocessing` gate

For `lang == eng || lang == fra`, Rev.AI is called with
`skip_postprocessing=true`. This suppresses Rev.AI's built-in punctuation
so that BA3's BERT utseg model handles sentence boundary detection. For all
other languages, Rev.AI post-processing is applied. Gate implemented in
`batchalign/inference/asr.py` — `_revai_request()`.

---

## `transcribe_s` vs `transcribe`

`transcribe_s` is not a separate CLI command. It is an internal command
variant triggered by `--diarization enabled`. Both share the same
`transcribe_pipeline.rs` orchestrator; the only difference is whether the
dedicated speaker stage runs.

---

## Testing

```bash
# Fast unit tests (no ML models)
make test

# Transcribe golden tests (real ASR models — only on net)
cargo nextest run --profile ml -E 'test(transcribe::)'

# Python ASR inference tests
uv run pytest batchalign/tests/test_asr.py -m golden
```

---

## Related developer documentation

- [Command Flowcharts: transcribe](../architecture/command-flowcharts.md#transcribe) — detailed runtime flowchart
- [ASR Token Pipeline](../architecture/asr-token-pipeline.md) — post-processing details
- [Cantonese Engines](../architecture/hk-cantonese-engines.md) — Tencent, Aliyun, FunASR
- [Number Expansion](../reference/number-expansion.md) — per-language Rust expansion
