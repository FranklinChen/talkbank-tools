# transcribe

**Status:** Current
**Last updated:** 2026-09-16 22:56 EDT

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

# Auto-detect the recording's language (one language for the whole file)
batchalign3 transcribe recording.wav -o out/ --lang auto

# Code-switched English/Spanish, with Rev.AI's multilingual model
batchalign3 transcribe bilingual.wav -o out/ --lang eng,spa

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
    engine_check -->|"whisper_rs, tencent, aliyun, funaudio or paraformer, qwen"| other_asr["Other engines\nprovider adapter pairs each unit\nwith its own timestamp"]

    rev_key --> rev_cache{"Validated raw Rev-evidence cache"}
    rev_cache -->|hit| rev_convert["Convert retained raw Rev transcript"]
    rev_cache -->|miss/forced refresh| rev_call["Authorized Rev language-ID/submit/poll"]
    rev_call --> rev_store["Validate + required durable commit"]
    rev_store --> rev_convert
    rev_convert --> asr_tokens

    whisper --> asr_tokens
    whisper_hub --> asr_tokens
    other_asr --> asr_tokens

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
        p1[1. Compound merging] --> p1check{lang=yue?}
        p1check -->|Yes| p2["2. Cantonese normalization\nonce per monologue\nOpenCC + domain replacements"]
        p1check -->|No| p3
        p2 --> p3
        p3[3. Multi-word splitting\nsplit tokens with spaces, interpolate timestamps]
        p3 --> p4[4. Number expansion\ndigits → word form]
        p4 --> p5[5. Long-turn splitting\nchunk at >300 words]
        p5 --> p6[6. Retokenization\npunctuation-based utterance splitting]
        p6 --> p7[7. Disfluency replacement\nfilled pauses + orthographic from per-language wordlists]
        p7 --> p8[8. N-gram retrace detection\nwrap repeated n-grams in `&lt;...&gt; [/]`]
    end

    postprocess --> speaker_apply{Dedicated speaker\nsegments present?}
    speaker_apply -->|Yes| project["Project segments onto timed ASR words\nby greatest summed overlap\nSplit chunks at speaker changes"]
    speaker_apply -->|No| utseg_check{"with_utseg?\ndefault: true"}
    project --> utseg_check

    utseg_check -->|Yes| run_utseg[process_utseg_with_evidence\nBERT-based re-segmentation]
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

Utterance segmentation also runs as a second pass over the built CHAT, and that
pass needs a segmenter. A language in the first list has one. A language in the
second does not, unless you pass `--utseg-fallback-stanza`, and **a run that
asks for utterance segmentation in such a language is refused when the job is
planned, before any ASR runs**. That matters because ASR is the expensive part:
the refusal used to happen at the very end of the pipeline, after the whole
transcription had been produced and paid for, and it named the worker's wire
format (`invalid utseg V2 result`) rather than the missing model. So a Spanish,
French, Japanese or Catalan transcription either authorizes the fallback:

```bash
batchalign3 transcribe corpus-spa/ -o out/ --lang spa --utseg-fallback-stanza
```

or is told immediately that it has no segmenter.

With `--lang auto` the language is not known until ASR has returned, so for
that case alone the same refusal necessarily happens after ASR rather than
before it.

For the FunASR engines (`funaudio`, `paraformer`), Chinese ASR tokens are
single characters (Latin words stay whole), each with its own timestamp, and
the provider's punctuation is not used as a boundary. Builds before 2026-09-15
paired Paraformer clauses with per-character timestamps, which mistimed whole
transcripts; re-run Paraformer transcripts made with them. Details:
[ASR token pipeline](../../architecture/asr-token-pipeline.md#provider-adapters-funasr-unit-admission).

For a supported language, normal transcription applies the utterance model
again after CHAT construction. This second pass can refine boundaries using
the completed main-tier context. Standalone `utseg` remains available for
re-segmenting an existing CHAT transcript without transcribing media again.

With `--wor`, a post-CHAT split that has complete timing for all partitioned
word tiers receives a main-tier timing bullet on every child, derived from
that child's own word span. If complete child timing is unavailable, no child
receives a main-tier bullet at all. The parent's bullet measures the whole
parent, so its start is where the first child began and its end is where the
last one finished, and writing it onto any one child would present a time
nobody measured as a measured one. The single exception is a split that kept
one child, where the parent's span is still exactly that child's. This is a
timing-preservation rule, not a claim that the model's segmentation is the only
acceptable CHAT segmentation.

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
| `--lang CODE` | `eng` | 3-letter ISO language code, `auto` for language auto-detection, or a code-switched pair such as `eng,spa` (primary language first; see [Code-switched recordings](#code-switched-recordings---lang-engspa)) |
| `--asr-engine NAME` | `rev` | ASR engine; see the table below. `--help` prints the same list, generated from the engines that exist, so neither can go stale. |
| `--asr-engine-custom NAME` |: | **Deprecated alias for `--asr-engine`**, still honoured so existing scripts keep working. Hidden from `--help`. |
| `--num-speakers N` | `2` | Speaker count passed to Rev.AI and to the dedicated diarizer. With `--diarization enabled` it must be 2 or more: a count of 1 is refused at submission, not obeyed. NOT a worker count; see `--workers`. No short flag, deliberately: see below. |
| `--auto-speakers` | off | Let the provider infer the speaker count instead. Rev.AI only: any other `--asr-engine` is refused before the job runs. Conflicts with `--num-speakers`. |
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
headers list only the speakers that actually occur in the output, so an
over-diarized transcript lists more of them and carries extra speaker tiers.
What is genuinely absent is the CAUSE. The requested count is recorded
nowhere in the file, so a reader seeing four speakers cannot tell an
over-diarized run from a session that really had four.

An UNDER-count is not absorbed either. The count is passed to Rev.AI and to
the dedicated diarizer, and the local `pyannote` and `nemo` engines produce
exactly the number they are given, so a count below the real number of
voices merges voices together. (An earlier version of this page said the
pipeline took `max(num_speakers, detected)`; no such step exists.) Only
Rev.AI can infer the count, with `--auto-speakers`.

**A count of 1 is refused, not obeyed.** With `--diarization enabled`, a
count of 1 asks a diarizer to separate the speakers of a recording asserted
to hold one, which is a contradiction rather than a request. The job is
refused at submission:

```text
diarization was requested with a speaker count of 1, which asks a diarizer to separate speakers in a recording asserted to have one. Pass the real count (2 or more); or omit diarization, for a single-speaker recording; or pass auto_speakers to have the count inferred, where the ASR engine supports it.
```

So with diarization the count is 2 or more, or it is omitted and the number
is detected (on Rev.AI, `--auto-speakers`). One is neither, and it is refused
rather than silently treated as detection. It used to be obeyed: the count
reached the diarizer, which returned a single track, which is why
`--diarization enabled` runs came back with one `PAR0`.

Speaker codes are also per recording. Each file is diarized on its own, so
`PAR0` in one transcript and `PAR0` in another need not be the same person,
even when a corpus has fixed participants; see
[diarize](diarize.md#output-format).

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
| `paraformer` | FunASR loading the Paraformer checkpoint: shorthand for `funaudio` with `funaudio_model=paraformer-zh`. Commonly wanted for Mandarin. An explicit `--engine-overrides '{"funaudio_model":"..."}'` wins over the implied checkpoint. Either way the transcript's `asr_model=` records what actually loaded: the alias resolves to the checkpoint this build pins, and a checkpoint it does not pin is loaded anyway and recorded at the revision the worker reports for it. |

```bash
# Mandarin with Paraformer.
batchalign3 transcribe Mandarin_mp3 -o out --lang zho --asr-engine paraformer
```

The engine NAME goes to `--asr-engine`. `--engine-overrides` takes a JSON
object of per-engine settings, and it is parsed while the command line is,
so a name passed to it (`--engine-overrides paraformer`) is refused before
anything runs, with `--asr-engine paraformer` named in the message rather
than a JSON syntax complaint.

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

Detection chooses ONE language for the whole file. It does not transcribe a
code-switched recording in two languages; `--lang eng,spa` below does.

---

## Code-switched recordings: `--lang eng,spa`

A pair declares that a recording mixes two languages. The first is the primary
language (the one an unmarked utterance is in, and the first `@Languages`
entry), the second the other language. Either order is accepted.

```bash
batchalign3 transcribe miami/ -o out/ --lang eng,spa   # English primary
batchalign3 transcribe miami/ -o out/ --lang spa,eng   # Spanish primary
```

What happens:

- **Rev.AI only, English/Spanish only.** The pair is sent as Rev.AI's
  multilingual English/Spanish model (`language: "en/es"`). Any other engine,
  and any other pair, is refused at submission, before anything is paid for.
  Every command other than `transcribe` refuses a pair.
- **Headers and provenance** declare both languages: `@Languages: eng, spa`
  and `lang=eng,spa` in the `[fc-ba3 transcribe | ...]` stamp.
- **No speaker count and no spoken-form switch.** Rev.AI's API refuses both
  `speakers_count` and `skip_postprocessing` for `en/es` (HTTP 400, checked
  against the live API on 2026-09-16), so neither is sent. Rev.AI's own
  speaker labels are unguided by `--num-speakers`, and its written forms
  (digits, `80%`) reach post-processing.
- **Numerals stay digits.** Nothing says which language `25` was spoken in,
  and writing it out in either would put words in the transcript the speaker
  may not have said, so `tengo 25 años` keeps `25`, and `80%` becomes `80`.
  The digits fail CHAT's word rules and are reported with the run's other
  refused words, for a human to transcribe.
- **Segmentation and morphosyntax run under the primary language.** A Spanish
  primary language has no TalkBank boundary model, so `--lang spa,eng` needs
  `--utseg-fallback-stanza` exactly as `--lang spa` does.
- **Utterances are not marked by language.** The transcript declares both
  languages but writes no `[- spa]` precodes, `@s` markers or code-switch
  spans; every utterance is in the primary language as far as the CHAT says,
  and filler, retrace and capitalization rules run under the primary.

---

## What gets created

A new `.cha` file per audio input (audio extension replaced: `foo.wav` →
`foo.cha`). Contains:

- a structured provenance `@Comment` (`[fc-ba3 transcribe | ...]`) plus a
  human-readable warning,
  `fc-ba3 <build identity>, ASR engine <engine>. Unchecked output of ASR model, DO NOT USE.`,
  carrying the build identity, the actual ASR engine name (with the models that
  produced the text in parentheses, in the same form `asr_model=` uses), and
  `DO NOT USE` for unchecked model output. A run that reported no models, such
  as one replaying legacy evidence, carries no parenthetical at all rather than
  naming what was requested. Re-transcribing replaces an earlier warning of
  ours, including the older `Batchalign <version>, ASR Engine <engine>.` form
- `@Languages`, `@Participants`, `@ID` headers
- Utterance lines with timing bullets
- `%wor` tier (if `--wor` is set)

No `%mor` or `%gra` tiers are created by `transcribe`. Run `morphotag`
afterwards if morphosyntactic analysis is needed.

---

## Gotchas

**Rev.AI `skip_postprocessing`:** BA3 sends `skip_postprocessing=true` for
English and Spanish, the languages Rev.AI's documentation lists for it. Which
languages get it is the option column of `REV_LANGUAGES` in
`crates/batchalign/src/types/revai_language.rs`, read by `rev_submit_options`
in `crates/batchalign/src/revai/asr.rs`. It is not sent for the `en/es`
multilingual model, which refuses it, nor for French or Portuguese, which the
live API's refusal message lists as accepting it but BA3 has never sent it
for. The flag is true because CHAT records spoken form (`"eighty percent"`,
`"seventeen year old"`); leaving it off causes Rev.AI to apply ITN and return
main-tier-illegal forms like `"80%"` / `"17-year-old"`. Where it is not sent,
BA3's post-processing writes numerals out in the transcript's language, or,
for a code-switched pair, keeps them as digits.

**A recording with no recognized words fails; it does not produce an empty
file.** If the ASR engine returns no words, or post-processing keeps no
utterance from the words it did return, or every token that reaches CHAT
assembly is a terminator or separator, the job fails and says which of those
three happened. Until 2026-09-16 the first of these wrote a transcript of
headers and nothing else and reported the job completed, which is
indistinguishable from a correct transcript of a silent recording. If the
recording does contain speech, the usual causes are the wrong `--lang` for the
audio or an engine that has no model for it.

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
