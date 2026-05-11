# Model Downloads and Caching

**Status:** Current
**Last updated:** 2026-05-11 11:35 EDT

## The contract

batchalign3 downloads every ML model it needs **automatically**, the first
time a command needs it. You never have to seed models, run setup scripts,
or remember which language pack lives where. The only error you should ever
see related to model downloads is one of these:

- "Failed to download …: network unreachable" — your machine can't reach
  the internet (or the upstream is down). Try again when you have network.
- "Failed to download …: disk full" — free some space and retry.
- "Failed to download …: HTTP 401/403" — the model requires authentication
  (rare; only one or two pyannote variants need this and BA3 ships with a
  default that doesn't).

If you see anything else — anything along the lines of "capability table is
unavailable", "resources.json could not be read", "model not found locally"
— that is a batchalign3 bug. file an issue on GitHub. You should not
have to think about model storage.

## What downloads, when, and roughly how big

Every download is **one-time**. After the first run, the model lives in your
local cache and the same command runs without any download.

| When you run | What downloads | Approximate size | Approximate first-run wait |
|---|---|---|---|
| `morphotag` (any language) | Stanza resource catalog (`resources.json`) | ~1 MB | 1–2 seconds |
| `morphotag` (first time for a language) | Stanza language pack for that language | 250–500 MB | 30 s – 2 min |
| `morphotag --retokenize` on a Cantonese file (`@Languages: yue`) | Nothing extra — PyCantonese is bundled | — | Instant |
| `morphotag --retokenize` on a Mandarin file (`@Languages: cmn`/`zho`) | Stanza Chinese tokenizer | ~200 MB | 30–60 s |
| `transcribe` (Whisper engine) | Whisper ASR model from HuggingFace | 0.5–3 GB depending on model size | 1–10 min |
| `align` (Whisper engine) | Whisper FA model from HuggingFace | ~3 GB | 3–10 min |
| `align` (Wave2Vec engine, default) | Wave2Vec MMS_FA bundle from torchaudio | ~1.2 GB | 1–5 min |
| `align --lang yue` (Cantonese FA) | Wave2Vec Cantonese model | ~1 GB | 1–5 min |
| `transcribe` (with diarization) | pyannote `talkbank/dia-fork` from HuggingFace | ~500 MB | 1–3 min |
| `translate` (Seamless engine) | SeamlessM4T from HuggingFace | ~2.4 GB | 2–8 min |
| `transcribe` (utterance segmentation, certain languages) | BERT utterance model from HuggingFace | ~400 MB | 1–3 min |

These sizes are ballpark. Real numbers depend on the upstream artifact and
your network speed.

## What you'll see on first run

Every download surfaces in your console, the TUI, the desktop app, and the
web dashboard at `http://<host>:8001/dashboard/jobs/<id>` — whichever UI
you're using. This is a deliberate UX commitment (see the [time
transparency principle](../architecture/time-transparency.md)): you should
always know what batchalign3 is doing and roughly how long it will take.

Example: a brand-new install running `batchalign3 morphotag input/ output/`
will show a sequence like:

```text
Downloading Stanza resource catalog (one-time, ~1 MB; future runs will be instant)…
Stanza resource catalog ready.
Downloading Stanza language pack for eng (en) (one-time, ~250–500 MB; future runs will use the local cache)…
Loading Stanza English…
Processing input/file1.cha
Processing input/file2.cha
…
```

A fresh `transcribe` run:

```text
Downloading openai/whisper-large-v3 for ASR (one-time, ~3 GB; future runs will use the local cache)…
Loading Whisper-large-v3 onto GPU…
Calling Rev.AI for utterance-timing recovery…
Processing recordings/r001.wav
…
```

If you see "Downloading X…" and the run sits there for a while, that is
expected — the download is running in the background. The libraries also
print their own progress bars to your terminal stderr.

After the first successful run, the same command on the same language runs
without any download — typically in seconds (for text NLP) to a few minutes
(for audio passes, dominated by inference, not loading).

## Where models are stored

batchalign3 uses each library's own cache. Locations vary by OS because each
library follows its own platform conventions:

| Library | macOS | Linux | Windows |
|---|---|---|---|
| Stanza (1.11+) | `~/Library/Caches/stanza/<resver>/resources/` | `~/.cache/stanza/<resver>/resources/` | `%LocalAppData%\stanza\<resver>\resources\` |
| HuggingFace (Whisper, Wave2Vec, SeamlessM4T, pyannote, BERT) | `~/.cache/huggingface/hub/` | `~/.cache/huggingface/hub/` | `%LocalAppData%\huggingface\hub\` |
| torchaudio (Wave2Vec MMS_FA bundle) | `~/.cache/torch/hub/torchaudio/` | `~/.cache/torch/hub/torchaudio/` | `%LocalAppData%\torch\hub\torchaudio\` |
| PyCantonese | Bundled in the package — no separate cache | (same) | (same) |

`<resver>` in the Stanza path is the resource-catalog version (e.g.
`1.11.0`), which Stanza bumps independently of the package version.

Combined cache size for a multi-language workflow can reach 10–30 GB.

## Customizing cache locations

Each library honors a standard environment variable for cache override.
Useful when you want to put caches on an external drive, a shared mount,
or a smaller SSD:

| Library | Environment variable |
|---|---|
| Stanza | `STANZA_RESOURCES_DIR` |
| HuggingFace | `HF_HOME` (controls all subpaths) or `HF_HUB_CACHE` (just the model hub) |
| torchaudio | `TORCH_HOME` (controls all torch hub caches) |

Example: offload everything to an external drive:

```bash
export STANZA_RESOURCES_DIR="/Volumes/External/caches/stanza"
export HF_HOME="/Volumes/External/caches/huggingface"
export TORCH_HOME="/Volumes/External/caches/torch"
batchalign3 daemon start
```

If you set these *before* the first model download, batchalign3 downloads
straight to the new location. If you set them *after* you've already
downloaded somewhere else, models will re-download — copy the existing
cache directories first to avoid that.

## Working offline

After the first successful run with internet access, batchalign3 works
fully offline for the same languages and engines. To enforce strictly-
offline behavior (and surface a clear error if any model is missing rather
than attempting a download), set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Stanza always tries the local cache first; no equivalent flag is needed.

In strict offline mode, the user-facing error for a missing model is along
the lines of "model X not in local cache; offline mode is enabled" —
actionable, distinct from the download-failed errors above.

## Pre-seeding for offline / air-gapped deployments

For deployments where the worker will run without internet (CI runners,
air-gapped fleets, conference demos), do the first download on a machine
with internet, then copy the cache directories to the offline target.

Stanza catalog + English pack:

```bash
python -c "import stanza; stanza.download('en')"
```

HuggingFace models (download to a known local path you can rsync):

```bash
python -c "
from huggingface_hub import snapshot_download
for repo in [
    'openai/whisper-large-v3',
    'talkbank/dia-fork',
    'facebook/hf-seamless-m4t-medium',
]:
    snapshot_download(repo)
"
```

Then copy the cache directories listed above to the offline machine, set
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, and run.

## Disk-space management

To free disk space by removing cached models (they will re-download next
use):

```bash
# macOS
rm -rf ~/Library/Caches/stanza/
rm -rf ~/.cache/huggingface/hub/
rm -rf ~/.cache/torch/hub/

# Linux
rm -rf ~/.cache/stanza/
rm -rf ~/.cache/huggingface/hub/
rm -rf ~/.cache/torch/hub/
```

Removing caches mid-job is safe: any in-flight download will continue, and
future jobs will re-download what they need. Removing caches while a daemon
is running will not crash the daemon — the next job that needs the missing
model will simply re-download it (and surface a download notification to
you, as documented above).

## Result caching (separate from model caching)

batchalign3 caches **audio-bound intermediate results** so repeated runs of
`align` or `transcribe` on the same file with the same settings do not redo
the expensive ASR / forced-alignment passes. Two task kinds are cached,
both per-utterance:

- `forced_alignment` — Wave2Vec word timings.
- `utr_asr` — full-file ASR result used for utterance-timing recovery.

The authoritative list of cached task kinds is in
`crates/batchalign/src/chat_ops/cache_key.rs::CacheTaskName`. Cache keys
include the engine version, language, and relevant inputs, so changing any
of those produces a fresh entry.

**Text NLP tasks are not cached.** Running `morphotag`, `utseg`,
`translate`, or `coref` twice on the same file runs the model twice. The
CLI accepts `--override-media-cache-tasks morphosyntax` for backward-
compatible scripting but emits a warning ("`batchalign3 does not cache
text NLP`") and ignores it.

To force re-computation of cached audio tasks (e.g., after a model update):

```bash
batchalign3 align --override-media-cache corpus/ -o output/ --lang eng
batchalign3 transcribe --override-media-cache recordings/ -o transcripts/ --lang eng
```

`--override-media-cache` clears all audio-task caches for the run. For
finer control, pass `--override-media-cache-tasks` with one or more of
`forced_alignment` / `utr_asr`.

The result cache lives at:

| OS | Path |
|---|---|
| macOS | `~/Library/Caches/batchalign3/cache.db` |
| Linux | `~/.cache/batchalign3/cache.db` |
| Windows | `%LocalAppData%\batchalign3\cache.db` |

You can `rm` it at any time; new runs will start a fresh cache.

## When something goes wrong

Three classes of error you might legitimately see, and what to do:

**Network unreachable.** "Failed to download Stanza catalog: network
unreachable" or "Failed to download model X: connection timeout". Check
your internet connection or proxy; retry. If you're on a corporate
network with a firewall, you may need to allow `https://huggingface.co`
and `https://raw.githubusercontent.com`.

**Disk full.** "Failed to write model file: no space left on device".
Free up space (see "Disk-space management" above) or move caches to a
larger drive (see "Customizing cache locations"). Then retry.

**Authentication.** "Failed to download X: HTTP 401/403". The model
requires a HuggingFace auth token. The standard batchalign3 install does
not need any — if you've configured a custom model that requires auth,
set `HF_TOKEN` in the daemon environment.

**What you should never see.** Errors mentioning "capability table",
"resources.json", "model not installed", or any internal-implementation
language. If you see one of these, batchalign3 is failing to download
something it should have downloaded automatically. File a bug
on GitHub and include the error message verbatim plus
the full command you ran. The team will fix the on-demand path; you
will not be asked to seed models manually.

## Related references

- [Time transparency UX principle](../architecture/time-transparency.md) —
  why downloads (and other slow operations) always surface to the UI.
- [Developer-facing model downloads doc](../developer/model-downloads-and-caching.md) —
  internals: load paths, cache invalidation, test strategy.
