# Model Downloads and Caching (Developer Reference)

**Status:** Current
**Last updated:** 2026-09-10 01:56 EDT

This page documents how batchalign3 downloads, caches, and verifies ML
models, the contributor-facing complement to the
[user-facing chapter](../user-guide/model-downloads.md). It is the
authoritative inventory of every model load site, every cache location, and
every download mechanism BA3 currently uses.

## The on-demand download contract

batchalign3 has one rule for ML models: **download on demand, transparently,
or surface a real error.** Concretely:

1. Every model family auto-downloads on first use through the upstream
   library's standard mechanism (Stanza's `DownloadMethod.REUSE_RESOURCES`,
   HuggingFace's `from_pretrained()`, torchaudio's `pipelines.MMS_FA.get_model()`).
2. No code in BA3 may opt out of these defaults. Specifically banned:
   `local_files_only=True`, `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`
   forced in BA3-controlled environment, `DownloadMethod.NONE`, or
   pre-flight existence checks that reject before the library would
   download. (One regression-gate test enforces this; see below.)
3. Any download that would block the worker for more than a second emits a
   `progress_v2` event with user-facing wording, propagated to every UI
   surface. See the [time transparency principle](../architecture/time-transparency.md).
4. A real failure (network, disk, auth) surfaces as a typed error the
   orchestrator can classify and the user can act on. Silent return-None
   on failure is the bug pattern this contract was written to prevent.

This contract was made explicit on 2026-05-06 after a fresh-install code
path silently failed: BA3 swallowed Stanza's `ResourcesFileNotFoundError`,
returned `None` from `get_cached_capability_table()`, and the Stanza
pre-flight gate translated the silent-None into "language not supported"
, misleading for a user whose Stanza catalog had simply never been
seeded. A single-host instance of that loop (orchestrator retry × worker
exit-1 × full Python traceback) generated multi-GB of `server.log` spam
per day.

## Inventory: every model load site

Source verified by reading code on 2026-05-06.

| # | Family | Load site | Library | Cache root |
|---|---|---|---|---|
| 1 | Stanza morphosyntax | `batchalign/worker/_stanza_loading.py:99` `load_stanza_models` | `stanza.Pipeline(download_method=REUSE_RESOURCES)` | Stanza `DEFAULT_MODEL_DIR` |
| 2 | Stanza utseg | `_stanza_loading.py:280` `load_utseg_builder` | (same) | (same) |
| 3 | Stanza Chinese retok | `_stanza_loading.py:235` `load_stanza_retokenize_model` | (same) | (same) |
| 4 | Stanza coref (lazy) | `batchalign/inference/coref.py:66-68` | `stanza.Pipeline(...)` | (same) |
| 5 | Whisper ASR | `batchalign/inference/asr.py:119` `load_whisper_asr` | `transformers.pipeline + WhisperProcessor.from_pretrained` | HF |
| 6 | Whisper FA | `batchalign/inference/fa.py:114` `load_whisper_fa` | `WhisperForConditionalGeneration.from_pretrained` + `WhisperProcessor.from_pretrained` | HF |
| 7 | Wave2Vec FA | `batchalign/inference/fa.py:198` `load_wave2vec_fa` | `torchaudio.pipelines.MMS_FA.get_model()` | torchaudio hub |
| 8 | Cantonese FA | `batchalign/inference/languages/cantonese/_cantonese_fa.py` `load_cantonese_fa` | `Wav2Vec2ForCTC.from_pretrained` | HF |
| 9 | SeamlessM4T translation | `batchalign/worker/_model_loading/translation.py::_load_seamless_translate` | `AutoProcessor.from_pretrained` + `SeamlessM4TModel.from_pretrained` | HF |
| 9b | NLLB-200 translation | `batchalign/worker/_model_loading/translation.py::_load_nllb_translate` | `AutoTokenizer.from_pretrained` + `AutoModelForSeq2SeqLM.from_pretrained` (`facebook/nllb-200-distilled-1.3B`, ~5 GB) | HF |
| 10 | pyannote diarization | `batchalign/inference/speaker.py:350` | `Pipeline.from_pretrained("talkbank/dia-fork")` | HF |
| 10b | Pyannote speaker embedding | `batchalign/inference/speaker_embedding.py::load_speaker_embedding_model` | `PretrainedSpeakerEmbedding(<pinned local ONNX path>)` | HF |
| 11 | NeMo speaker (fallback) | `batchalign/inference/speaker.py` (NeMo branch) | `EncDecSpeakerLabelModel.from_pretrained(...)` | NeMo cache |
| 12 | BERT utterance | `batchalign/models/utterance/infer.py:120-128` | `AutoTokenizer.from_pretrained` + `BertForTokenClassification.from_pretrained` | HF |
| 13 | PyCantonese | (bundled) |, | (none, wheel) |

Cache roots resolve to OS-specific paths via each library's own logic. See
the [user-facing chapter](../user-guide/model-downloads.md) for the table
of OS-resolved paths.

### Stanza `DEFAULT_MODEL_DIR` (1.11+)

Resolves via `os.getenv('STANZA_RESOURCES_DIR', os.path.join(USER_CACHE_DIR, 'resources'))`
in `stanza/resources/common.py:38-41`. `USER_CACHE_DIR` is the platform
cache plus a versioned subdirectory:

- macOS: `~/Library/Caches/stanza/<resver>/resources/`
- Linux: `~/.cache/stanza/<resver>/resources/`
- Windows: `%LocalAppData%\stanza\<resver>\resources\`

The historical `~/stanza_resources/` from older Stanza versions is no
longer used and any references to it in BA3 docs are bugs to fix.
`<resver>` is the resource-format version (e.g., `1.11.0`), independent of
the package version (e.g., `1.11.1`).

### HuggingFace cache resolution (current)

Order: `HF_HUB_CACHE` env > `HF_HOME` env > default
(`~/.cache/huggingface/hub` on Unix, `%LocalAppData%\huggingface\hub` on
Windows). The legacy `TRANSFORMERS_CACHE` is no longer consulted by
current `huggingface_hub`; do not reintroduce it.

## Catalog bootstrap (Stanza-specific)

Stanza ships its package code without `resources.json`. The catalog must be
downloaded once before any language pack can be resolved. BA3 does this
automatically:

- `_stanza_capabilities.py:get_cached_capability_table()` calls
  `build_stanza_capability_table()`, which calls
  `stanza.resources.common.load_resources_json()`.
- On `ResourcesFileNotFoundError` (a subclass of `FileNotFoundError`),
  `_bootstrap_and_retry()` calls `stanza.resources.common.download_resources_json()`,
  emits start/complete `progress_v2` events, and rebuilds the table.
- A real download failure raises typed `StanzaCatalogDownloadError`; the
  orchestrator should classify this as non-retryable at the worker-
  bootstrap layer (filed separately).
- `ImportError` on `import stanza` is the one legitimate silent-None path:
  it means the worker venv lacks the package, which is a deploy-config
  error, not a recoverable miss.

The pre-flight capability table itself remains the right thing for
rejecting languages Stanza does not actually have processors for (e.g.
`que`). It MUST NOT block on missing-but-downloadable resources. That
distinction is what the catalog bootstrap exists to enforce.

## User-visible download notifications

Every download site emits a `progress_v2` event so the user sees what's
happening. The shared helper lives at
`batchalign/worker/_progress.py`:

- `emit_download_event(stage, user_message, request_id=None, size_bytes_estimate=None)`
 , generic, used for non-HF downloads (Stanza catalog, Stanza language
  packs, torchaudio bundles).
- `emit_hf_download_if_missing(model_id, kind, request_id=None)`: probes
  the HuggingFace cache via `try_to_load_from_cache`; emits only when the
  model is genuinely about to download. Wraps every `from_pretrained()`
  call.

Sample: every HF load site looks like

```python
from batchalign.worker._progress import emit_hf_download_if_missing

emit_hf_download_if_missing("openai/whisper-large-v3", kind="ASR")
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", ...)
```

The wrapping is cheap (one cache probe), idempotent for cached models
(probe returns hit, no event emitted), and safe under failure (probe
exceptions log debug-level and emit anyway, a false-positive notification
is a much smaller UX cost than a silent multi-minute wait).

User-message wording must convey four things: what's downloading, the
approximate size, that it's a one-time cost, and that future runs will be
instant. Size hints for the largest models are tabulated in
`_progress.py` `_HF_SIZE_HINTS_GB`; expand the table when adding new
families.

## Audit gates (regression prevention)

A static check in `batchalign/tests/test_progress_audit.py` (planned)
asserts that no new code reintroduces opt-outs. Specifically, it greps for:

- `local_files_only=True` in any `from_pretrained()` call
- `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` set inside BA3-controlled
  environment construction (test environments may set them externally,
  which is fine)
- `DownloadMethod.NONE` in any Stanza `Pipeline()` call
- pre-flight existence checks that raise before the library would download

If any future PR needs an exception (e.g., an offline-test fixture), it
must be opt-in via a code-path-specific flag, not a default.

## Pipeline-result caching (orthogonal to model caching)

batchalign3 caches **audio-task evidence** in a tiered cache so repeated
`align`, selected `transcribe` stages, and standalone `diarize` do not repeat
expensive inference.
This is unrelated to ML-model caching: result evidence has semantic keys and
revision identities of its own.

| Layer | Storage | TTL | Location |
|---|---|---|---|
| Hot | `moka` in-memory | Per-process lifetime | RAM |
| Cold | SQLite | Persistent | `~/.cache/batchalign3/cache.db` (Linux), `~/Library/Caches/batchalign3/cache.db` (macOS), `%LocalAppData%\batchalign3\cache.db` (Windows) |

Cached task kinds are enumerated in
`crates/batchalign/src/chat_ops/cache_key.rs::CacheTaskName`: forced-alignment
projection, raw forced-alignment worker evidence, normalized UTR ASR, raw Rev
transcript evidence, raw speaker evidence, and derived speaker segments. Cache
keys include the task's relevant combination of:

- Evidence schema and algorithm/preparation revision
- Language code
- Engine/model revision
- Relevant per-task inputs

### Engine identity comes from the selected worker

The engine-version namespace is resolved from the exact typed worker route,
not from whichever worker first populated the pool's availability snapshot.
`WorkerKey` owns target, language, and engine recipe. Command dispatch obtains
that worker, completes any lazy `ensure_task`, queries its live capabilities,
and only then constructs the `EngineVersion` used by `PipelineServices` and
the tiered cache.

Lazy-profile keys retain engine selection. This is required for correctness,
not only cache hygiene: a task-only key could load Wave2Vec once, report
`already_loaded` to a later Whisper request, and let concurrent requests change
process-global model state. Engine-specific keys make those states
unrepresentable; host worker permits and idle eviction still bound memory.

Raw evidence remains payload-validated on replay. That is defense in depth,
not a substitute for an honest namespace. The SQLite migration
`20260831000000_fa_raw_evidence_engine_namespace.sql` repairs this historical
boundary once when a cache is opened. A schema-2 payload owns the exact
selected-worker version and can therefore repair a contradictory database
label. A schema-1 payload owns only the requested engine family; when that
family contradicts the stored version family, the migration copies the exact
row into `cache_quarantine` with a stable reason and removes it from live
lookup instead of inventing producer provenance. Correctly labelled schema-1
rows remain replayable under their explicit `legacy_cache_namespace` origin.
The cache-stats command reports quarantined counts by reason without counting
them as live hits.

Row 10b downloads NOTHING new. It resolves the `embedding` node of the same
manifest row 10 uses, by the same exact Hub commit, through the same
pinned-artifact loader. The difference is that it constructs the embedding
model STANDALONE instead of as part of a `SpeakerDiarization` pipeline. That
matters operationally rather than only architecturally: the diarization
pipeline class unconditionally loads a PLDA calibration artifact from a gated
repository (see the [diarize page](../user-guide/commands/diarize.md)), so
diarization needs a Hugging Face token where embedding does not. A machine with
no Hugging Face account can embed and cannot diarize.

Local Pyannote has an additional cross-language identity rule. The JSON
manifest at `batchalign/inference/local_pyannote_model.json` is the single
owner of the exact pipeline, segmentation, and embedding Hub commits and
artifact SHA-256 digests. Python downloads the pinned revisions, copies each
artifact into a private directory while hashing its bytes, and refuses any
digest mismatch before loading the pipeline. Segmentation and embedding loaders
receive these verified local paths; the worker retains the snapshots for lazy
backend reads. This admits the packaged trusted model bytes, not arbitrary
checkpoint files, and does not make general pickle loading safe. Rust
hashes the identical packaged bytes into `SpeakerEvidenceModelRevision`.
Changing any graph node therefore invalidates raw speaker evidence without a
second hand-maintained version constant or a drift-detection test.

Text NLP tasks (`morphotag`, `utseg`, `translate`, `coref`) are NOT
cached, running them twice runs the model twice.

### Raw versus derived revisions

Raw paid-service evidence and locally derived projections have different
identities. Changing a speaker normalization algorithm bumps only
`SpeakerNormalizationRevision`; the retained provider response remains usable.
Changing the provider request, prepared media bytes, backend/model revision, or
raw schema changes the raw key. Rev evidence follows the same rule: preserve
the provider-shaped transcript, then rerun local conversion independently.

Rev's provider-media boundary is explicitly typed. Production constructs
`PreparedRevProviderMedia` from source bytes, recording their BLAKE3 digest,
the revisioned preparation recipe, and a normalized upload filename. The raw
request key also includes the multipart MIME and request-policy revision. On a
typed cache miss, `RevAsrInferenceAuthorization` is consumed into exactly one
`AuthorizedRevEvidenceRun` and one private commit permit. Immediately before
upload, the run rereads the file and refuses `ProviderMediaDrift` if its bytes
no longer match the keyed digest. Both language identification and
transcription receive the same owned verified byte buffer and presentation.
For auto language, those are two intentional Rev requests inside one typed
evidence run; explicit-language transcription needs only the latter.

Request-identity revision 2 changes the earlier key because revision 1 did not
fully identify the provider-visible multipart presentation. Storage schema 3
is separate: the HTTP boundary reads transcript application-body bytes,
requires strict UTF-8 JSON, and retains that exact sequence including unknown
provider fields without changing a revision-2 request key. It never assigns
exact fidelity after lossy or normalizing text decoding. Existing schema-2
envelopes therefore remain replayable as `legacy_typed_projection`; new
responses are `exact_provider_json`. `CachePolicy::RequireCache` fails closed
on a true request miss and never turns a storage migration into a service call.

The present recipe is `SourceBytesLegacyAudioMpegV1`: source bytes, normalized
`provider-media.<extension>` filename, digest-derived metadata, and the
historical `audio/mpeg` multipart MIME. Do not interpret that name as a quality
endorsement. A future WAV, FLAC, MIME-correct, padding-retained, or other recipe
must be a distinct enum variant with deterministic tests and a distinct key.

`RevAsrEvidenceResolution::trace()` uses a private trace seed captured by the
resolver from the exact request. Callers therefore cannot pair one request's
media and cache identity with another resolution's cache outcome. The method
adds the explicit downstream projection revision and produces schema-2
`RevAsrEvidenceTrace`. `DebugDumper::dump_rev_evidence()` atomically persists
it for Rev transcribe and Rev-backed UTR runs when `--debug-dir` is enabled,
using the collision-resistant full-input identity. Serialization or
durable-write failure is a file error, not a best-effort log message.

The transcribe pipeline receives the Rev inference boundary as a
`RevAsrEvidenceInference` capability. This is dependency injection without an
authorization escape hatch: the trait method accepts only the consuming
`AuthorizedRevEvidenceRun` created behind `resolve_rev_asr_evidence()`. A
pipeline-level counting fake can consequently prove cold-versus-replay
behavior through CHAT construction, while production still cannot submit a
request directly from an untyped media path or Boolean cache miss.

`RevAsrEvidenceTrace` distinguishes causal origin from semantic projection.
Cold and replayed runs have different cache outcomes by design. Their typed
semantic projection consists of the complete request trace seed, retained
transcript fidelity, and named downstream projection revision. Tests compare
that projection instead of deleting arbitrary JSON fields or incorrectly
requiring the two causal records to be byte-identical.

Transcribe names the ASR projection revision
`rev-transcript-to-asr-response-v1`; UTR names its narrower timed-word
projection `rev-transcript-to-utr-asr-response-v1`. Partial UTR windows use a
stable raw-cache-key suffix so multiple evidence records cannot overwrite one
another.

`--require-media-cache` is the experiment guard. It selects
`CachePolicy::RequireCache`; raw Rev/speaker misses fail before inference
authorization, and unresolved FA groups cannot construct
`FaInferenceAuthorization`. `--override-media-cache` is the opposite policy:
it intentionally refreshes and replaces evidence. The CLI and HTTP admission
reject a job that asks for both.

## Test strategy

### Unit tests (no network, no models)

Bootstrap behavior under mocked filesystem and Stanza APIs lives in
`batchalign/tests/test_stanza_capabilities.py`. The three load-bearing
cases:

- `test_bootstrap_downloads_catalog_when_missing`: `resources.json`
  absent + download succeeds → populated table returned.
- `test_bootstrap_raises_typed_error_on_download_failure`: absent +
  download fails → `StanzaCatalogDownloadError`.
- `test_stanza_not_installed_returns_none`: `ImportError` → `None`
  (unchanged silent-None path, the only legitimate one).

These run in the default `pytest` profile (no `-m golden` needed); they
mock all I/O.

### Golden tests (real models, network on first run)

Tests that load real ML models are marked `@pytest.mark.golden` and
excluded from the default `pytest` run:

```bash
uv run pytest -m golden                # Python golden tests
cargo test -p batchalign --features ml-golden --test ml_golden         # Rust ML golden tests
```

Models download automatically on first run. Subsequent runs use the cache.
First-run download is slow (minutes for Stanza, longer for Whisper).

### Fresh-install integration test

`batchalign/tests/test_fresh_install_stanza_bootstrap.py` nukes the
Stanza cache, walks the bootstrap path, and asserts the catalog
auto-downloads. This is the canonical regression gate for the
on-demand contract: if it fails, BA3 has reintroduced a download
opt-out somewhere.

### OOM protection in golden tests

On machines with < 128 GB RAM, the `conftest.py` guard forces golden tests
to run sequentially (`-n 0`) even if the default `pytest.ini` specifies
parallel workers. Each Stanza model instance uses 2-5 GB, parallel
workers on a 64 GB machine OOM-crash. The guard cannot be bypassed; it
fires per-test inside xdist workers via an autouse fixture.

PyCantonese tests run in the default suite because PyCantonese is bundled
(no download needed) and fast (~3s for all segmentation tests).

### What to expect on first run

| Test suite | First-run download | Subsequent runs |
|---|---|---|
| `uv run pytest` (default) | PyCantonese: 0s (bundled) | < 1s |
| `uv run pytest -m golden` | Stanza English: ~2 min, Stanza Chinese: ~2 min | < 30 s |
| `cargo test -p batchalign --features ml-golden --test ml_golden` | Stanza + Whisper: ~5-10 min | < 2 min |

## Adding a new model load site

1. Identify the upstream library's auto-download API (`from_pretrained`,
   `Pipeline`, `get_model`, etc.). Use it as-is. Do not pre-flight-check.
2. Add a `progress_v2` emit immediately before the load:
   - HuggingFace: `emit_hf_download_if_missing(model_id, kind=...)`.
   - Stanza language pack: extend the helper in `_stanza_loading.py` (or
     copy its shape).
   - Other libraries: use `emit_download_event(stage, user_message)`.
3. Add a size-hint entry to `_HF_SIZE_HINTS_GB` if the model is > 100 MB,
   so the user sees a useful estimate.
4. Update the [user-facing chapter](../user-guide/model-downloads.md)
   table with the new family + size + first-run wait estimate.
5. Update this page's inventory table.
6. Add a golden-marked test that exercises a fresh download path.

## Related references

- [User-facing model-downloads chapter](../user-guide/model-downloads.md).
- [Time transparency principle](../architecture/time-transparency.md).
- The contract enforcement code: `batchalign/worker/_stanza_capabilities.py`,
  `batchalign/worker/_progress.py`, `batchalign/worker/_protocol.py`.
- Bootstrap regression tests: `batchalign/tests/test_stanza_capabilities.py`.
