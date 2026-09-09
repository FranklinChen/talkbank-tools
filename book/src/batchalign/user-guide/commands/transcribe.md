# transcribe

**Status:** Current
**Last updated:** 2026-09-07 07:04 EDT

Create a new CHAT transcript from audio files using automatic speech
recognition (ASR). Produces `.cha` files alongside or in a separate output
directory. **Never modifies the input audio.**

---

## Quick start

```bash
# Transcribe a single recording, output alongside input
batchalign3 transcribe interview.wav

# Transcribe all audio files in a directory
batchalign3 transcribe recordings/ -o transcripts/ --lang eng

# Auto-detect language (useful for bilingual/code-switched audio)
batchalign3 transcribe bilingual.wav -o out/ --lang auto

# Transcribe with paid pyannoteAI Precision-2 diarization, the default
batchalign3 transcribe interview.wav -o out/ --asr-engine whisper --diarization enabled

# Keep all audio local and use the TalkBank-pinned Pyannote model
batchalign3 transcribe interview.wav -o out/ --diarization enabled --speaker-engine pyannote

# Use the remote server
batchalign3 --server http://your-server:8001 transcribe corpus/ -o out/ --lang eng
```

Dedicated diarization defaults to the pyannoteAI Precision-2 cloud model. Set
its API key in the environment:

```bash
export BATCHALIGN_PYANNOTE_API_KEY="your-key"
```

`PYANNOTE_API_KEY` and `BATCHALIGN_PYANNOTE_KEY` are also accepted. For
compatibility with existing installations, BA3 also reads
`engine.pyannote.key` from the `[diarize]` section of `~/.batchalign.ini`.
The environment variables take precedence.

pyannoteAI receives the recording through its temporary-media API. Its use can
incur account charges, so confirm the account plan and the recording's data-use
or IRB rules before running it. Job output is not written into the API key
configuration.

The local `--speaker-engine pyannote` alternative uses the public, ungated
TalkBank-pinned `talkbank/dia-fork` pipeline plus its pinned segmentation and
embedding dependencies, downloaded anonymously on first use. It runs
inference locally and does not use the pyannoteAI API key. `--speaker-engine
nemo` is a second local alternative that avoids the paragraph below entirely.

**`--speaker-engine pyannote` also fetches one UNPINNED, currently GATED
artifact: a PLDA calibration model.** `pyannote.audio`'s pipeline class loads
it unconditionally during construction, and the released config does not
override it, so the class's own default applies: the gated
`pyannote/speaker-diarization-community-1` repository. A machine with no
accepted terms and no Hugging Face token fails on first use naming that
repository. The fix is a token, checked in this order: `~/.batchalign.ini`
`[auth] hf_token`, then Hugging Face's own resolution (`HF_TOKEN`, or the
token saved by `hf auth login`); accepting the repository's terms at
<https://huggingface.co/pyannote/speaker-diarization-community-1> is required
either way. Full detail: [diarize](diarize.md#local-model-download-todays-truth-including-a-gated-dependency).

A Hugging Face token is NOT relevant to the default command shown above,
which uses `--speaker-engine pyannote-ai` and never reaches this local path.

The standalone [`diarize`](diarize.md) command is intended for producing
anonymous `.turns.json` evidence for an existing transcript. It defaults to
local Pyannote but can explicitly select paid pyannoteAI Precision-2; both
surfaces share the same raw/derived speaker-evidence cache. Integrated
diarized transcription projects speaker evidence onto
timed ASR words before utterance segmentation and CHAT construction; it does
not require a later `chatter rediarize` pass.

### Rev and speaker evidence caching

Rev.AI transcription now caches the raw provider-shaped transcript before BA3
token conversion and post-processing. A normal warm run checks and validates
that evidence before submitting anything to Rev, so it avoids another Rev
call. Keeping raw monologues, elements, punctuation, timings, confidence, and
resolved language also lets BA3 post-processing experiments replay the same
service output locally.

With `--diarization enabled`, BA3 durably caches both the backend-shaped
speaker evidence and the normalized turns used by the transcribe pipeline.
For pyannoteAI, retained raw evidence includes the completed job ID, full
output object, and warning. Repeating the same recording with the same speaker
backend, expected speaker count, preparation recipe, and model revision
replays validated turns without calling the diarization backend again. Copies
and renames of byte-identical recordings share the entry.

Raw evidence and derived turns have separate revision identities. A later BA3
release can change how provider output is converted to speaker intervals and
recompute those intervals locally from the retained response. That kind of
algorithm experiment does not upload the audio or incur another pyannoteAI
inference charge.

This is particularly important for the paid `pyannote-ai` default. A corrupt
entry fails visibly and does not fall through to another paid call. Concurrent
identical files in one server are coalesced so only the first miss runs
inference.

For a replay experiment that must not turn a missing Rev or speaker entry into
a service call, require the cache explicitly:

```bash
batchalign3 --require-media-cache transcribe interview.wav -o out/ \
  --asr-engine rev --diarization enabled
```

A raw-evidence miss then fails visibly before service inference is authorized.
If raw speaker evidence exists but normalized turns do not, BA3 may still
recompute those turns locally. The flag applies to cache-backed stages; it does
not cache or suppress ordinary local ASR engines.

To run a deliberate fresh experiment, use the global override:

```bash
batchalign3 --override-media-cache transcribe interview.wav -o out/ \
  --diarization enabled
```

The fresh results replace the matching cache entries. The override can incur
new Rev and pyannoteAI charges. Other ASR engines are not yet cached in
ordinary `transcribe` runs. See [Caching](../caching.md) for exact invalidation
rules and limitations.

---

## Pipeline

```mermaid
flowchart TD
    start([transcribe invoked]) --> resolve[Resolve audio file]
    resolve --> ensure_wav[ensure_wav: convert if needed]

    ensure_wav --> diarize_check{--diarization?}
    diarize_check -->|"enabled"| transcribe_s["Command: transcribe_s\nASR + dedicated speaker relabeling\nRev or Whisper"]
    diarize_check -->|"auto/disabled\n(default)"| transcribe_m["Command: transcribe\nDefault path\nRev labels used directly when present"]

    transcribe_s --> engine_check
    transcribe_m --> engine_check

    engine_check{--asr-engine?}
    engine_check -->|whisper| whisper[Whisper local ASR]
    engine_check -->|whisper_hub| whisper_hub["HF Whisper fine-tune\n(per-language model_id)"]
    engine_check -->|rev| rev_key["Hash provider media + Rev request semantics"]
    engine_check -->|"whisperx, whisper_oai"| refused["Refused: engine not implemented"]

    rev_key --> rev_cache{"Validated raw Rev-evidence cache"}
    rev_cache -->|hit| rev_convert["Convert retained raw Rev transcript"]
    rev_cache -->|miss/forced refresh| rev_call["Authorized Rev language-ID/submit/poll"]
    rev_call --> rev_store["Validate + required durable commit"]
    rev_store --> rev_convert
    rev_convert --> asr_tokens

    whisper --> asr_tokens
    whisper_hub --> asr_tokens

    asr_tokens["Raw ASR tokens\nword + start_s + end_s + optional speaker + confidence"]
    asr_tokens --> convert["convert_asr_response()\nGroups tokens by speaker label"]
    convert --> dedicated_check{"--diarization enabled?"}
    dedicated_check -->|No| postprocess
    dedicated_check -->|Yes| speaker_key["Hash source bytes + semantic request"]
    speaker_key --> speaker_cache{"Validated derived-segment cache"}
    speaker_cache -->|hit| retain_check
    speaker_cache -->|derived miss| speaker_raw{"Validated raw-evidence cache"}
    speaker_raw -->|hit| speaker_normalize["Versioned local normalization"]
    speaker_normalize --> speaker_store_derived["Commit derived segments"]
    speaker_store_derived --> retain_check
    speaker_raw -->|raw miss/forced refresh| speaker_v2["execute_v2(task=speaker)\nprepared audio → backend evidence\npyannoteAI Precision-2 by default"]
    speaker_v2 --> speaker_store["Validate + commit raw evidence\nthen derived segments"]
    speaker_store --> retain_check{"--debug-dir?"}
    retain_check -->|Yes| retain_turns["Write same-job canonical speaker turns\nwith typed backend provenance"]
    retain_check -->|No| postprocess
    retain_turns --> postprocess

    subgraph postprocess ["Rust post-processing: process_raw_asr()"]
        direction TB
        p1[1. Compound merging] --> p2[2. Multi-word splitting\nsplit tokens with spaces, interpolate timestamps]
        p2 --> p3[3. Number expansion\ndigits → word form]
        p3 --> p3check{lang=yue?}
        p3check -->|Yes| p4["4. Cantonese normalization\nOpenCC + domain replacements"]
        p3check -->|No| p5
        p4 --> p5[5. Long-turn splitting\nchunk at >300 words]
        p5 --> p6[6. Retokenization\npunctuation-based utterance splitting]
        p6 --> p7[7. Disfluency replacement\nfilled pauses + orthographic from per-language wordlists]
        p7 --> p8[8. N-gram retrace detection\nwrap repeated n-grams in `&lt;...&gt; [/]`]
    end

    postprocess --> speaker_apply{Dedicated speaker\nsegments present?}
    speaker_apply -->|Yes| project["Project segments onto timed ASR words\nby greatest summed overlap\nSplit chunks at speaker changes"]
    speaker_apply -->|No| utseg_check{"with_utseg?\ndefault: true"}
    project --> utseg_check

    utseg_check -->|Yes| run_utseg[process_utseg\nBERT-based re-segmentation]
    utseg_check -->|No| mor_check{"with_morphosyntax?\ndefault: false"}

    run_utseg --> build_chat["build_chat → ChatFile AST\nHeaders, participants, %wor tiers"]
    utseg_check -->|No| build_chat
    build_chat --> mor_check
    mor_check -->|Yes| run_mor[process_morphosyntax\nPOS + lemma + depparse]
    mor_check -->|No| merge_check

    run_mor --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| output

    merge --> output[Serialize → .cha output]
    output --> done([Output .cha file])
```

---

## Utterance boundary detection

`transcribe` always does utterance splitting before CHAT output is written.
There are two paths:

- `eng`, `cmn`, `zho`, `yue`: dedicated pre-CHAT utterance models
- all other languages, punctuation-based splitting in Rust

For a supported language, normal transcription applies the utterance model
again after CHAT construction. This second pass can refine boundaries using
the completed main-tier context. Standalone `utseg` remains available for
re-segmenting an existing CHAT transcript without transcribing media again.

With `--wor`, a post-CHAT split that has complete timing for all partitioned
word tiers receives a main-tier timing bullet on every child, derived from
that child's own word span. If complete child timing is unavailable, BA3 keeps
the original enclosing bullet on the last child only; it does not invent
intermediate timings. This is a timing-preservation rule, not a claim that the
model's segmentation is the only acceptable CHAT segmentation.

---

## Options

### Path options

| Option | Meaning |
| --- | --- |
| `PATHS...` | Audio files (`.mp3`, `.mp4`, `.wav`) or directories |
| `-o`, `--output DIR` | Output directory for new `.cha` files |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Write `.cha` files alongside the audio inputs |

### ASR and language options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | `eng` | 3-letter ISO language code, or `auto` for language auto-detection |
| `--asr-engine NAME` | `rev` | ASR engine; see the table below. `--help` prints the same list, generated from the engines that exist, so neither can go stale. |
| `--asr-engine-custom NAME` |: | **Deprecated alias for `--asr-engine`**, still honoured so existing scripts keep working. Hidden from `--help`. |
| `--num-speakers N` | `2` | Expected number of speakers. NOT a worker count; see `--workers`. No short flag, deliberately: see below. |
| `--diarization {auto,enabled,disabled}` | `auto` | Dedicated speaker diarization stage (`auto` = disabled) |
| `--speaker-engine {pyannote-ai,pyannote,nemo}` | `pyannote-ai` when enabled | Paid pyannoteAI Precision-2 cloud diarization, or an explicit local engine |
| `--wor` / `--nowor` | `--nowor` | Include or suppress the `%wor` word-timing tier |
| `--merge-abbrev` | off | Merge abbreviations in the output |
| `--utseg-fallback-stanza` | off | Opt in to the legacy Stanza constituency-parser fallback for utterance segmentation when no TalkBank BERT model is configured for `--lang`. Default refuses substitution. See [utseg → Language support](utseg.md#language-support). |


### Why `--num-speakers` has no short flag

`-n` was a short form for it until 2026-08-19. It was removed because `-n`
reads as a job or worker count nearly everywhere it appears, `pytest -n` most
of all, and this CLI has its own `--workers` beside the `make -j` and
`xargs -P` conventions. A caller reaching for parallelism was therefore
reconfiguring DIARIZATION, silently.

That is not a hypothetical. On 2026-08-19 a re-transcription of four sessions
was submitted as `-n 4` meaning four workers. Four speakers would have
over-diarized those sessions against the 361 siblings in the same delivery,
which were built with two.

**How visible would that have been?** Less than nothing, and more than an
earlier draft of this page claimed. The run SUCCEEDS, and no warning is
printed. The effect is visible if you look: `@Participants` and the `@ID`
headers are rebuilt from the speakers the diarizer actually returned, so an
over-diarized transcript lists more of them and carries extra `*SPK:`
prefixes. What is genuinely absent is the CAUSE. The requested count is
recorded nowhere in the file, so a reader seeing four speakers cannot tell an
over-diarized run from a session that really had four.

Note also that an UNDER-count is absorbed: the pipeline takes
`max(num_speakers, detected)`. Only over-counts change the result.

`-n` is now a hard error, which is the point. A caller who meant parallelism
goes looking and finds `--workers`; a caller who meant speakers finds
`--num-speakers`. Both land where they intended, and neither is silently
misread. On every command but `diarize` the flag defaults to 2, so most
callers never type either; `diarize` has no default, because omitting it and
letting the engine auto-detect is the recommendation there.

Note that `--workers` has its own trap, documented on the flag itself: it
applies to NEW daemons, and does not change the parallelism of a daemon that
is already running and being reused.

#### ASR engines

Every engine is reachable through the single `--asr-engine` flag. This list is
generated from the engine set itself, as is the one `--help` prints.

| Name | Notes |
| --- | --- |
| `rev` | Rev.AI cloud ASR. The default. |
| `whisper` | Local Whisper. |
| `whisper_hub` | HuggingFace Whisper fine-tune by model id. See [`whisper-hub-asr.md`](../../reference/whisper-hub-asr.md). |
| `whisperx` | **Not implemented.** Accepted as a name and refused at submission; nothing here runs WhisperX. |
| `whisper_oai` | **Not implemented.** Accepted as a name (`whisper-oai` is the historical spelling) and refused at submission; nothing here calls the OpenAI Whisper API. |
| `whisper_rs` | Rust-native whisper.cpp, run in process. |
| `tencent` | Tencent Cloud ASR. |
| `aliyun` | Aliyun ASR. |
| `funaudio` | FunASR / SenseVoice. Local, no credentials, no network. |
| `qwen` | Qwen3-ASR. Local. |
| `paraformer` | FunASR loading the Paraformer checkpoint: shorthand for `funaudio` with `funaudio_model=paraformer-zh`. Commonly wanted for Mandarin. An explicit `--engine-overrides '{"funaudio_model":"..."}'` wins over the implied checkpoint. |

```bash
# Mandarin with Paraformer.
batchalign3 transcribe Mandarin_mp3 -o out --lang zho --asr-engine paraformer
```

---

## Speaker labeling and segmentation

This is the most common source of confusion with `transcribe`.

**Rev.AI (default engine):** Rev.AI returns speaker labels as part of its ASR
response. These labels are **always** applied, you get multi-speaker output
without passing `--diarization enabled`. Passing `--diarization enabled`
explicitly makes dedicated diarization authoritative and ignores Rev's speaker
projection. The default dedicated engine is pyannoteAI Precision-2.

**Whisper-based engines** (`--asr-engine whisper`, `whisper_hub`, `whisper_rs`):
these engines produce no speaker labels. Without `--diarization enabled`, all
utterances are attributed to a single default speaker. Pass
`--diarization enabled` to run a dedicated speaker stage that assigns speaker
identities.

**`--diarization auto`** (the default) = disabled dedicated stage. Equivalent
to BA2's `--nodiarize`. The BA2 help text claiming Rev.AI ignored `--diarize`
was stale, the actual BA2 `transcribe_s` pipeline wiring ran the dedicated
stage.

Dedicated diarization is integrated before utterance segmentation. BA3 first
post-processes the timed ASR words, then assigns each timed word to the speaker
with the greatest summed diarization overlap and splits prepared chunks where
that label changes. The language-specific utterance model therefore sees the
speaker boundaries and cannot merge across them. Untimed punctuation inherits
the nearby timed label. Timed words in gaps take the nearest dedicated segment
label. Once dedicated evidence exists, no word can re-enter the unrelated ASR
label space and create a phantom participant. Diagnostics report contested
words, unattested words, and inserted speaker boundaries.

### Retaining the exact diarization turns used by transcription

For research, replay, or merge-pipeline evaluation, pass `--debug-dir PATH`
with `--diarization enabled`. In addition to the other debug artifacts, BA3
writes one `<audio-stem>.turns.json` file containing the exact
dedicated segments used to build that transcript. The artifact uses the same
deterministic `PAR` coordinate system as the generated CHAT and records typed
backend provenance, including `batchalign3:pyannote_ai:precision-2` for the
cloud default.

BA3 also writes a versioned `*_speaker_evidence.json` sidecar. It records the
source digest, preparation revision, backend and expected-speaker request,
model and normalization revisions, raw and derived cache keys, whether the run
replayed derived evidence, re-normalized raw evidence, or inferred after a
miss, the named segment-projection revision, and a versioned digest and count
of the exact validated segments. It excludes machine-local source paths and
credentials.

When this artifact is requested, failure to create or write it fails the file
instead of silently discarding the evidence. With a remote server, `PATH` is
on the server host; the CLI sends an absolute path.

Keep both sidecars, the run manifest, and generated CHAT together: the causal
record explains where the evidence came from, while the turns artifact is the
exact normalized projection consumed downstream.

For Rev ASR, the same `--debug-dir` run also writes a versioned,
collision-resistant `*_rev_evidence.json` sidecar. It records the source and
provider-media digests, preparation recipe, exact multipart presentation,
language and speaker request, model/request revisions, raw cache key, cache
outcome, and local projection revision. This sidecar is also fail-closed when
requested and excludes the Rev credential and machine-local source path.

For languages with a TalkBank utterance-boundary model, BA3 also writes a
versioned `*_pre_chat_utseg_evidence.json` sidecar. Normal model-backed
transcription writes a separate `*_post_chat_utseg_evidence.json` sidecar for
the second pass over completed main tiers. Keeping the phases separate prevents
an experiment from confusing boundaries over timed ASR chunks with later
boundaries over main-tier words.

Each item records the exact input words and applied group assignments. A
boundary-model result additionally records its model ID and revision plus one
evidence state per word: raw action, action after adjacency policy, and
sentence-end probability, or an explicit normalization-omission or
short-input state. Constituency-tree projection and compatibility assignments
without model evidence are separate source variants. BA3 refuses a worker
result whose assignments or evidence do not exactly parallel the input words.
As with the Rev and speaker causal records, an enabled utseg evidence write is
atomic and fail-closed.

The current boundary model is lexical and contextual. It does not itself
receive waveform energy, pause duration, pitch, diarization overlap, or CHAT
retrace structure. The sidecar makes its contribution inspectable so research
code can compare it with those signals without rerunning the model.

---

## `--lang auto` behavior

With `--asr-engine whisper`, `--lang auto` omits the language parameter from
Whisper's generation kwargs, letting the model detect the spoken language from
the audio. The multilingual `openai/whisper-large-v3` model is always used
with `auto`: language-specific fine-tuned models are bypassed because they
are trained for a single language.

With Rev.AI, `--lang auto` submits a true auto-language request to the Rev.AI
API. Note that Rev.AI auto-detect and explicit `--lang eng` can produce
different punctuation, diarization, and turn boundaries from the provider.

---

## What gets created

A new `.cha` file per audio input (audio extension replaced: `foo.wav` →
`foo.cha`). Contains:

- a structured provenance `@Comment` plus a human-readable warning carrying
  the Batchalign version, actual ASR engine name, and `DO NOT USE` for unchecked
  model output
- `@Languages`, `@Participants`, `@ID` headers
- Utterance lines with timing bullets
- `%wor` tier (if `--wor` is set)

No `%mor` or `%gra` tiers are created by `transcribe`. Run `morphotag`
afterwards if morphosyntactic analysis is needed.

---

## Gotchas

**Rev.AI `skip_postprocessing`:** For English and Spanish (the only
languages Rev.AI's API documents the parameter as supporting), Rev.AI
is called with `skip_postprocessing=true`. The hint table is in
`crates/batchalign/src/revai/preflight.rs::skip_postprocessing_hint`,
which matches Rev.AI's own 2-letter codes `"en" | "es"` and returns
`None` for everything else. The flag is true because CHAT records
spoken form (`"eighty percent"`, `"seventeen year old"`); leaving it
off causes Rev.AI to apply ITN and return main-tier-illegal forms
like `"80%"` / `"17-year-old"`. For languages outside the en/es
support pair, no flag is sent (the parameter is a no-op there per
Rev.AI's docs), and BA3's downstream post-processing handles
spoken-form normalization.

**`--server` requires server-visible audio.** With `--server`, the server
resolves audio paths on its own filesystem. Paths valid on your machine must
also be reachable from the server, or you must use a shared media mount.

**Memory on developer machines.** Each Whisper model instance uses 2-15 GB.
For large corpus runs (more than a handful of files or >1 GB audio total),
prefer a dedicated server with substantial RAM (via `--server`) over a
developer laptop, and always pass `--workers 1` for local smoke tests.

---

## Related documentation

- [Rev.AI Integration](../rev-ai.md), API key setup, engine behavior
- [Cantonese Engines](../cantonese-processing.md), Tencent, Aliyun, FunASR engines
- [Utterance Segmentation](../../reference/utterance-segmentation.md), post-ASR BERT utseg
- [Command I/O: transcribe](../../reference/command-io.md#2-transcribe), I/O patterns and mutation behavior
- [Command Flowcharts: transcribe](../../architecture/command-flowcharts.md#transcribe), full architecture flowchart
- [ASR Token Pipeline](../../architecture/asr-token-pipeline.md), ASR post-processing details
