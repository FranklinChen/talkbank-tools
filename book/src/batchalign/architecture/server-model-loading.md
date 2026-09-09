# Server Model Loading and Caching

**Status:** Current
**Last updated:** 2026-09-07 07:04 EDT

This document describes every ML model loaded by batchalign3 workers,
when each model is loaded into memory, and how results are cached.

## How Models Are Loaded

The Rust server spawns Python worker processes keyed by `(target, lang)`.
Targets are either:

- a released infer task such as `infer:morphosyntax` or `infer:asr`
- or a test-echo worker bootstrapped for one infer task without loading models

Each worker loads its own models on first use. Workers are managed
by a **WorkerPool** (`crates/batchalign/src/worker/pool/`) with a
configurable idle timeout (default 10 minutes) and automatic crash
restart.

For a multi-file job the runner **pre-scales** its workers before file
dispatch begins, so the batch pays one cold start rather than one per file. A
startup `warmup` of a configured command list existed until 2026-07-30; it had
been inert on real servers since 2026-03-26 and was removed.

---

## Per-Command Model Inventory

### `morphotag` (task string: `morphosyntax`)

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/morphosyntax.py` | Stanza pipeline (tokenize, pos, lemma, depparse, mwt) | stanza / HF Hub | 300-500 MB per language | First file for each language (lazy per-language dict) | Yes |

**Internal caching:** Per-language `stanza.Pipeline` dict in the worker state.
A single worker handles all languages without reloading.

**Result caching:** SQLite utterance cache. Key = `BLAKE3(words + lang + "|mwt")`,
gated by Stanza version. Stores final `%mor`/`%gra` strings.

---

### `align` (task string: `fa`)

The server auto-chains forced alignment + UTR + disfluency + retrace.

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/fa.py` (Whisper FA) | `openai/whisper-large-v2` | HF Hub | ~3 GB | Worker startup (immediate) | Yes |
| `inference/asr.py` (UTR) | `talkbank/CHATWhisper-en-large-v1` (eng) or `openai/whisper-large-v2` (other) | HF Hub | ~3 GB | First audio file (lazy) | Yes |
| Rust (disfluency) | None (rule-based data files) | local | negligible | N/A | No |
| Rust (retrace) | None (Rust n-gram) | local | negligible | N/A | No |

**Alternative:** Wave2Vec FA (`inference/fa.py`) uses `torchaudio.pipelines.MMS_FA`
(~1.6 GB, loaded at startup, from PyTorch Hub, not HF Hub).

**Result caching:**
- Forced alignment: SQLite. Key = `BLAKE3(audio identity + time window + words + gap-healing policy + engine)`.
- UTR: SQLite. Key = `BLAKE3(realpath + filesize)`. Protected from pruning.

---

### `transcribe` (server-owned composition over `asr`)

Current CLI default engine is Rev.AI. Alternate ASR engines are selected with
`--asr-engine whisper`, `--asr-engine whisper_hub`, or
`--asr-engine whisper_rs`. (`whisperx` and `whisper_oai` are accepted names
with no implementation; submitting either is refused.) The server auto-chains
disfluency + retrace.
For languages with a dedicated utterance model (`eng`, `cmn`, `zho`, `yue`),
transcribe also runs pre-CHAT utterance segmentation before CHAT assembly.

| Module / Engine | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| Rust `crates/batchalign/src/revai/asr.rs`: Rev (default) | Rev.AI HTTP client only | local + remote API | negligible local memory | per-file server dispatch | No |
| `inference/asr.py`: Whisper | `openai/whisper-large-v3` + optional BertUtteranceModel | HF Hub | ~3 GB + ~400 MB | Worker startup (immediate) | Yes |
| Rust (disfluency) | None | local | negligible | N/A | No |
| Rust (retrace) | None | local | negligible | N/A | No |

**BertUtteranceModel languages:** Only loaded when `resolve("utterance", lang)`
returns a model name. Currently: `eng` (`talkbank/CHATUtterance-en`),
`cmn` / `zho` (`talkbank/CHATUtterance-zh_CN`), `yue` (Cantonese-specific
model).

**Result caching:** Raw provider-shaped Rev.AI evidence is cached and replayed
after strict validation. Ordinary non-Rev ASR engines are not yet cached and
run inference again.

---

### `transcribe_s` (server-owned composition over `asr`)

`transcribe_s` now follows the same server-owned transcribe pipeline as
`transcribe`. The low-level task has no CLI command literally named `speaker`,
but it has two product surfaces: integrated diarized transcription and the
standalone `diarize` command. When the selected ASR
backend already returns usable speaker labels (for example Rev.AI or the
Cantonese provider adapters), Rust keeps those labels on the default path. When
`--diarize` is explicitly requested, Rust also composes the low-level `speaker`
infer task, receives raw diarization segments, and projects them onto timed ASR
words before utterance segmentation and CHAT assembly. That projection splits
prepared chunks at speaker changes even on top of Rev-labeled output.

The default dedicated diarization backend is pyannoteAI Precision-2. Its worker
adapter performs the typed `PreparedWav` to `UploadedMedia` to
`SubmittedDiarizationJob` to `CompletedDiarizationJob` lifecycle and requests
exclusive diarization for ASR reconciliation. Local Pyannote and NeMo remain
explicit alternatives. Local Pyannote loads lazily on the first request in a
worker process and is then reused within that process.

**Result caching:** Integrated diarized transcription and standalone
`diarize` share the same validated raw-evidence and derived-turn cache. A
normal warm run can therefore replay evidence without another paid or local
diarization call. Standalone defaults to local Pyannote but can explicitly
select pyannoteAI Precision-2 or NeMo; the backend and optional/known speaker
count are part of the evidence identity.

---

### `translate` (task string: `translate`)

| Module / Backend | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/translate.py`: Google (default) | None (Google Translate API) | remote | N/A | N/A | No |
| `inference/translate.py`: Seamless | `facebook/hf-seamless-m4t-medium` | HF Hub | ~1.2 GB | Worker startup (immediate) | Yes |

**Result caching:** SQLite utterance cache. Key = `BLAKE3(text + src_lang + tgt_lang)`.

---

### `utseg` (task string: `utterance`)

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/utseg.py` | `BertUtteranceModel` for `eng` / `cmn` / `zho` / `yue`; otherwise Stanza pipeline (tokenize, pos, lemma, constituency) | HF Hub / stanza | ~400 MB for BERT model or 300-500 MB per Stanza language | First batch (lazy factory) | Yes |

**Result caching:** SQLite utterance cache. Key = `BLAKE3(text + lang)`.

---

### `coref` (task string: `coref`)

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/coref.py` | Stanza tokenizer + `ontonotes-singletons_roberta-large-lora` | stanza / HF Hub | ~500 MB | First file (lazy) | Yes |

English only. **Result caching:** None.

---

### `benchmark` (server-owned composition over `asr`)

Same engines as `transcribe` plus a Rust-side WER step:

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `crates/batchalign-transform/src/benchmark.rs` | None (Rust Hirschberg DP alignment via the allowlisted `dp_align::align` call site) | local | negligible | N/A | No |

---

### `opensmile` (task string: `opensmile`)

| Module | Model | Source | Size | Loaded When | HF Hub |
|--------|-------|--------|------|-------------|--------|
| `inference/opensmile.py` | None (C++ feature extraction) | local | negligible | Worker startup | No |

Feature sets: `eGeMAPSv02`, `GeMAPSv01b`, `ComParE_2016`, `eGeMAPSv01b`.

**Result caching:** None (produces CSV output).

---

## Device Placement

All torch-based inference modules auto-detect the compute device at load time:

| Priority | Device | Notes |
|----------|--------|-------|
| 1 | CUDA | If `torch.cuda.is_available()` and not `--force-cpu` |
| 2 | MPS | macOS Metal (Apple Silicon). Used on Apple Silicon server/client machines when available |
| 3 | CPU | Fallback |

Stanza manages its own device internally (typically CPU).

## Speaker engine credentials

The default cloud engine reads `BATCHALIGN_PYANNOTE_API_KEY`,
`BATCHALIGN_PYANNOTE_KEY`, or `PYANNOTE_API_KEY`, in that order. It also accepts
`engine.pyannote.key` in the `[diarize]` section of `~/.batchalign.ini` for
compatibility. This is a worker-owned credential path and audio is uploaded to
pyannoteAI.

### Hugging Face downloads for the local Pyannote engine

The released local speaker engine loads three PINNED artifacts:
`talkbank/dia-fork`, `talkbank/seg-fork-3.0`, and
`hbredin/wespeaker-voxceleb-resnet34-LM`. All three repositories are public
and ungated, and a worker downloads them anonymously; none needs `hf auth
login`, `HF_TOKEN`, accepted model terms, or a pyannoteAI API key.

**A fourth, UNPINNED artifact is fetched behind those three, and it is
currently GATED.** `pyannote.audio`'s `SpeakerDiarization` pipeline class
loads a PLDA calibration artifact unconditionally during construction,
regardless of the pinned config's clustering choice; the released config does
not override it, so the class's own default applies, which is the gated
`pyannote/speaker-diarization-community-1` repository. A worker with no
accepted terms and no Hugging Face token fails on first use naming that
repository, mapped by `batchalign.inference._model_access_errors` to a typed
`ModelAccessDeniedError` (Rust: `ProtocolErrorCodeV2::ModelAccessDenied` /
`ServerError::ModelAccessDenied` / `FailureCategory::ModelAccessDenied`,
never `Validation`). The remedy is a Hugging Face token in
`~/.batchalign.ini` `[auth] hf_token` (checked before Hugging Face's own
`HF_TOKEN`/`hf auth login` resolution, via
`batchalign.inference.pyannote_local.resolve_huggingface_hub_token`), after
accepting the repository's terms at
<https://huggingface.co/pyannote/speaker-diarization-community-1>. User-facing
detail: [diarize](../user-guide/commands/diarize.md#local-model-download-todays-truth-including-a-gated-dependency).

The repository names are not the runtime identity. The packaged
`batchalign/inference/local_pyannote_model.json` manifest owns an exact
40-hex Hugging Face commit and required artifact for every node in the graph.
Python validates the manifest before downloading and rewrites the pipeline's
transitive model references to those pinned artifacts. Rust hashes those same
manifest bytes into the raw speaker-evidence model revision. A moving Hub head
therefore cannot change inference while retaining an old cache identity, and
the Python and Rust sides cannot drift through duplicated version constants.

Release-candidate verification on 2026-08-31 exercised the packaged manifest
through the real Pyannote loader, not only mocks: the loaded pipeline retained
the manifest's pinned segmentation revision, resolved the embedding from that
revision's snapshot path, and completed local inference over one second of
prepared silent 16 kHz PCM with the expected empty segment result. This is a
runtime wiring check, not a diarization-quality claim.

This differs from some upstream Pyannote pipelines whose publishers gate their
repositories. If an operator deliberately replaces the TalkBank-pinned model
with such a custom model, that operator must provide whatever Hugging Face
authentication and terms acceptance the custom repository requires. That
credential is not part of the default Batchalign3 deployment.

Once a model is downloaded, it is cached on disk at `~/.cache/huggingface/`
and does not re-download on subsequent loads.
