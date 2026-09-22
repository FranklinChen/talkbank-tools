# Caching

**Status:** Current
**Last updated:** 2026-09-22 14:10 EDT

## What gets cached

Batchalign caches **only audio-task results**:

| Analysis | Cached? |
|----------|---------|
| Forced alignment word timings (`align`) | Yes |
| ASR results for utterance timing recovery (`align`'s UTR pre-pass) | Yes |
| Dedicated speaker evidence (`transcribe --diarization enabled` or standalone `diarize`) | Yes |
| Media conversion (`.mp4`/`.m4a` → `.wav`) | Yes |
| Raw Rev.AI transcript evidence (`transcribe`, `benchmark`, Rev-backed `align` UTR) | Yes |
| Other ordinary ASR output (`transcribe`) | No |
| Morphosyntax (`morphotag`) | **No**: always recomputed |
| Utterance segmentation (`utseg`) | **No**: always recomputed |
| Translation (`translate`) | **No**: always recomputed |
| Coreference (`coref`) | **No**: always recomputed |
| OpenSMILE features (`opensmile`) | No |
| AVQI scores (`avqi`) | No |

The text-NLP cache that previously covered `morphotag`, `utseg`, and
`translate` was **removed** after a benchmark on a 15,748-file corpus
showed it was about 25× slower than just re-inferring (6-16% hit rate;
2,500 ms SQLite lookup beat ~100 ms inference savings). See the
architecture page on Caching for the detailed reasoning.

In practice: a re-run of `morphotag` on the same corpus takes the
same time as the first run. A re-run of `align` on the same audio is
much faster. A repeat `transcribe --diarization enabled` run with the
same speaker settings reuses the exact normalized speaker turns that
the first run consumed, instead of calling the diarization backend
again. BA3 also retains backend-shaped evidence separately, so a changed
local normalization algorithm can derive new turns without repeating
inference.

For Rev.AI transcription, BA3 reads the transcript endpoint as bytes, requires
strict UTF-8 JSON, and stores that exact application response before converting
its monologues and elements to BA3 tokens. It does not use lossy or
encoding-normalizing text decoding. Unknown provider fields are retained for
future projections. A warm run therefore avoids both Rev submission and
polling, and later post-processing experiments can replay the same provider
transcript evidence locally.

### Dedicated speaker evidence

The speaker cache applies to the dedicated diarization stage of `transcribe`
and to standalone `diarize`, for all three speaker engines (`pyannote-ai`,
`pyannote`, and `nemo`). It is especially useful with the paid pyannoteAI
service.

The key includes:

- a BLAKE3 digest of the full media-source bytes;
- the canonical audio-preparation recipe revision;
- the selected speaker backend;
- the expected speaker count;
- the speaker-model revision; and
- the stored evidence schema version.

Paths and modification times are deliberately excluded. Renaming or copying
an unchanged recording therefore reuses its speaker evidence. Re-encoding the
recording changes its bytes and causes a miss even if it sounds identical.

BA3 stores two different artifacts:

1. **Raw inference evidence.** For pyannoteAI this is the completed provider
   job ID, complete output object, and optional warning. Local Pyannote and
   NeMo retain their backend-specific segment evidence.
2. **Derived speaker segments.** These are the sorted millisecond intervals
   consumed by transcript speaker projection. Their identity includes both
   the raw-evidence fingerprint and a separate normalization-algorithm
   revision.

Changing only BA3's normalization algorithm invalidates the second artifact,
not the first. BA3 re-normalizes the retained provider response locally and
does not upload audio or submit another paid pyannoteAI job. Changing the
model, speaker count, backend, audio bytes, or preparation recipe changes the
raw identity and therefore requires inference.

BA3 validates schema versions, fingerprints, backend provenance, provider
job identity, speaker labels, interval direction, and ordering. A missing raw
entry permits inference; a corrupt raw or derived entry fails the file instead
of being treated as a miss and silently causing another billable call.
Concurrent identical requests in one BA3 server are serialized: the first
miss performs and commits inference, while followers wait and then replay the
result.

BA3 also rereads the source after a speaker-cache miss and verifies that its
bytes still match the digest used for the cache decision. Speaker inference is
prepared from that verified in-memory copy. If another process replaces the
media between lookup and inference, the file fails instead of running a paid
job under the wrong cache identity.

### Raw Rev.AI transcript evidence

Normal `transcribe`, `benchmark`, and Rev-backed `align` UTR runs check durable
raw Rev evidence before submitting anything to Rev. Only a missing entry, or
an explicit `--override-media-cache`, authorizes submission. Corrupt evidence
fails closed without a service call, and concurrent identical requests are
coalesced. Concurrent identical forced refreshes also share the first fresh
commit rather than each issuing a sequential paid call.

The key includes the full bytes of the provider-visible inference media, media
preparation recipe, normalized upload filename, multipart MIME type, requested
language, expected speaker count, Rev request-policy revision, provider/model
alias, and request-identity revision. The stored envelope contains the resolved
language plus the exact admitted transcript response JSON bytes. It does not
store credentials or temporary job IDs.

Before a paid call, BA3 rereads the prepared source and verifies that the exact
bytes still have the digest used for the cache decision. If the file changed
between preparation and submission, the run fails instead of uploading
different bytes under the old key. The authorization is then consumed into one
evidence-inference run plus a separate commit permit; the inference capability
cannot be cloned into repeated runs. An auto-language run intentionally
contains both Rev language identification and transcription requests.

Request-identity revision 2 names this stronger provider presentation. Entries
from revision 1 do not satisfy the new keys. Storage schema 3 adds exact JSON
retention without changing the revision-2 request key: schema-2 entries remain
replayable and are explicitly traced as `legacy_typed_projection`, while new
provider responses are traced as `exact_provider_json`. Replay-only mode never
turns either storage migration into a paid call.

For controlled Rev transcribe and Rev-backed align experiments,
`--debug-dir PATH` additionally writes versioned `*_rev_evidence.json` causal
sidecars. Each records the keyed media and
multipart presentation, request identity, `replayed` versus fresh-miss outcome,
raw evidence key, transcript fidelity, and deterministic projection revision
without exposing the credential or local source path.

Dedicated-speaker transcribe runs likewise write
`*_speaker_evidence.json`. This joins the source digest, request/model
semantics, raw and derived cache identities, cache outcome, normalization
revision, segment-projection revision, segment count, and a versioned digest
of the exact normalized timing/label projection. The companion `.turns.json`
holds those normalized segments in the canonical review format.

BA3's regression suite exercises this at the complete Rust transcribe-pipeline
boundary, not only at the cache row. It closes and reopens SQLite, replays the
same retained Rev response, and requires identical final CHAT plus identical
ASR-response debug output with no second inference call. The Rev causal
sidecars intentionally differ in exactly one meaning: the cold run records a
fresh missing-key inference and the warm run records replay. Their media,
request, retained-evidence, and projection identities must remain equal.

The old batch pre-submission shortcut has been removed because it submitted
paid jobs before a cache hit could be known. Cold misses currently fan out
through BA3's normal per-file worker limit. Reintroducing wider parallel
submission is a performance follow-up; it must consume the same typed miss
authorization and cannot restore the old unguarded optional-job-ID path.

Other ASR engines remain uncached in ordinary `transcribe` runs. Align's UTR
ASR also keeps its older normalized-result cache. If that derived entry is
missing, Rev-backed UTR can re-project the durable raw transcript without a
service call. A corrupt normalized UTR entry fails closed instead of silently
falling through to inference. The legacy Rev pre-submission path no longer
exists.

## How to guarantee cache-backed stages do not infer

Use `--require-media-cache` for replay-only experiments:

```bash
batchalign3 --require-media-cache transcribe recordings/ -o output/ \
  --asr-engine rev --diarization enabled

batchalign3 --require-media-cache align corpus/ -o output/
```

For every cache-backed stage reached by the command, a reusable hit is
required. Missing raw Rev or speaker evidence fails the file before an
inference authorization can be constructed, so the miss cannot become a Rev
or pyannoteAI call. Forced alignment likewise refuses to send missing groups
to its worker. Existing clean `%wor` evidence can still satisfy an FA group.

Raw and derived evidence remain separate. If normalized speaker turns are
missing but the backend-shaped speaker response exists, BA3 derives and stores
new turns locally. Rev-backed UTR can similarly rebuild its normalized UTR
entry from retained raw Rev evidence; if the raw entry is also missing, the
raw-evidence gate refuses the provider call.

Forced alignment also keeps raw and derived layers. On a normal hit, BA3
prefers the admitted worker response and reruns the current local timing
projection; an admitted versioned derived envelope is the fallback when raw
evidence is absent or refused. Both envelopes prove the requested engine,
selected-worker version, semantic group key, and word cardinality. Historical
bare timing vectors are refused and treated as misses because they cannot
prove whether the result came from the requested engine or an unversioned
fallback. This means experiments with Rust-side timing interpretation can
reuse identified direct model work automatically.

A live Wave2Vec-to-Whisper fallback is intentionally not cached today. It is
valid for the current output and is recorded in the debug trace, but the Wave
request's version namespace does not identify the effective Whisper model.
Persisting that response would make later replay ambiguous, so a future run
repeats the fallback. `--override-media-cache` and
`--override-media-cache-tasks forced_alignment` bypass both FA layers and
therefore request fresh model inference.

When a current build first opens an older cache, it also removes from live
lookup any legacy raw FA row whose requested engine contradicts the model
family in its stored namespace. Those rows did not retain an exact producer
version and cannot be safely reused or relabeled. Their exact stored bytes are
retained in the database's `cache_quarantine` table for audit. This cleanup can
turn an apparent historical hit into an honest miss;
`--require-media-cache` still refuses that miss without running inference.
`batchalign3 cache stats` reports the quarantine total and its stable reason
counts separately from reusable entries.

This flag is not a general offline, no-network, or zero-compute mode. Ordinary
non-Rev ASR output is not cached, so Whisper, Tencent, Aliyun, FunAudio, or Qwen
transcription can still run its configured inference path, including a network
service where that backend uses one. OpenSMILE and AVQI are outside the
analysis cache. The guarantee is specifically that a
missing entry at a cache-backed boundary cannot authorize inference.

`--require-media-cache` is mutually exclusive with
`--override-media-cache` and `--override-media-cache-tasks`: one run cannot
both require existing evidence and request fresh evidence.

## What invalidates the cache

| What changed | What re-runs | What stays cached |
|---|---|---|
| Edited the transcript words | FA (per-group cache key includes text) | UTR ASR (only depends on audio) |
| Re-recorded or replaced the audio | FA, UTR ASR, Rev evidence, speaker evidence | (n/a, audio is the cache key) |
| Changed the language code | UTR ASR and Rev evidence | (other corpora's entries) |
| Changed expected speaker count | Rev evidence and speaker evidence | FA and UTR ASR |
| Changed speaker backend | Speaker evidence | FA, UTR ASR, and Rev evidence |
| Changed only the speaker normalization algorithm | Derived speaker segments | Raw speaker inference evidence |
| Upgraded batchalign or an identified model revision | Affected audio evidence | Entries from unchanged engines/models |

Cache keys hash the inputs relevant to each task. FA and UTR use the legacy
path/mtime/size `AudioIdentity`; Rev and speaker evidence use a true digest of
the inference-media bytes. Rev also keys provider-visible presentation, so
copies and renames share results only when their normalized upload extensions
match. Engine or model revision strings are stored alongside each entry.

pyannoteAI currently exposes the `precision-2` model alias, but not an
immutable backend build hash. BA3 scopes cloud evidence to that alias and its
own evidence schema. If the provider changes the implementation behind the
same alias and you want fresh evidence, use `--override-media-cache`. The
local Pyannote and NeMo identifiers likewise include their configured model
identity and the BA3 package version; floating external model revisions remain
a reason to force a refresh during controlled experiments.

## How to force fresh results

Use the `--override-media-cache` global flag:

```bash
batchalign3 --override-media-cache align corpus/ -o output/

# Force and store fresh Rev and dedicated speaker evidence. This may incur charges.
batchalign3 --override-media-cache transcribe recordings/ -o output/ \
  --diarization enabled
```

This skips all applicable cache lookups, forcing fresh inference. New results
replace the matching entries and are stored for future runs. With
Rev ASR or `--speaker-engine pyannote-ai`, this can make new paid service calls
even when reusable evidence exists.

The narrower `--override-media-cache-tasks` flag accepts `forced_alignment`,
`utr_asr`, `rev_asr_evidence`, and
`speaker_diarization_raw_evidence`. This permits a controlled transcribe run to
refresh Rev while replaying speaker evidence, or the reverse, instead of
repeating both paid boundaries.

Use this when you suspect cached results are wrong, or after manually
updating model files outside of a normal batchalign upgrade.

## UTR ASR results after an upgrade

UTR ASR results are stored under a namespace naming the timing-recovery engine
AND the models it ran, for example
`utr-asr-v1:whisper_utr:whisper|asr=openai/whisper-large-v3@06f233fe...`.
Earlier builds stored them under the forced-alignment engine's version instead,
and then under the recovery engine's name alone. The first `align` run on this
build therefore misses those older UTR ASR entries once, recomputes them, and
stores them under the new namespace; later runs hit as before.
Forced-alignment entries are unaffected, because their namespace did not
change: it is still exactly the forced-alignment engine name the worker
reports.

Naming the models is what makes an upgrade safe rather than merely noticed.
Upgrading a recovery model now lands in a different namespace, so results
produced by the previous weights are never reused for the new ones. Nothing
reads the older rows: a namespace move makes them unreachable by construction,
so there is no compatibility path to keep true.

This costs one recompute, not two. The namespace already moved once in this
release, and the model identity was folded into that same move deliberately.

With `--require-media-cache`, that one recompute is refused like any other
miss for Whisper and Tencent recovery. Rev recovery can still rebuild the
entry from retained raw Rev evidence without a new provider call.

## Where the caches are stored

| Cache | macOS default | Linux default |
|---|---|---|
| Analysis cache DB | `~/Library/Caches/batchalign3/cache.db` | `~/.cache/batchalign3/cache.db` |
| Media conversion cache | `~/Library/Application Support/batchalign3/media_cache/` | `~/.local/share/batchalign3/media_cache/` |

The analysis cache is a single SQLite database file. The media cache
stores converted WAV artifacts for inputs such as `.mp4` and `.m4a`. Its key
is a hash of every byte of the source, so a cache hit still reads the whole
source file and skips only the conversion; on a network mount that read is a
full transfer. See [Network and Transfer Costs](network-costs.md).

For isolated runs or testing, you can relocate them with environment
variables:

```bash
export BATCHALIGN_ANALYSIS_CACHE_DIR=/tmp/ba-analysis-cache
export BATCHALIGN_MEDIA_CACHE_DIR=/tmp/ba-media-cache
```

`BATCHALIGN3_ANALYSIS_CACHE_DIR` is accepted as an alias for the analysis
cache setting. If both spellings are set, the canonical
`BATCHALIGN_ANALYSIS_CACHE_DIR` value wins. Set the variable before starting a
server: a client process cannot relocate the cache owned by an already-running
server.

## How to clear the cache

Use the built-in cache command:

```bash
batchalign3 cache stats          # See cache size and entry count
batchalign3 cache clear --yes    # Clear the cache
```

`cache stats` and `cache clear` operate on both the analysis cache and
the media conversion cache.

Or delete the `cache.db` file and/or the media-cache directory directly.

To selectively refresh without clearing everything, use
`--override-media-cache` on specific runs instead, old entries for
other corpora remain available.

## Old text-NLP cache entries

If you used batchalign before the text-NLP cache was removed, your
`cache.db` may still contain old `morphosyntax_v*`, `utseg_v*`, and
`translate_v*` rows. Those are dead weight, they're never read
anymore. Run `batchalign3 cache clear --yes` (or `rm -f
~/Library/Caches/batchalign3/cache.db*`) to reclaim the disk space.
