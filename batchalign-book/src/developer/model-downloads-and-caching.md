# Model Downloads and Caching

**Status:** Current
**Last updated:** 2026-03-23 11:35 EDT

## Overview

batchalign3 uses several ML models that are downloaded on first use and cached
locally. Developers need to know what gets downloaded, where it's stored, and
when cache invalidation happens — both for day-to-day work and for the test
suite.

## Model Inventory

| Library | Models | Trigger | Approximate Size | Cache Location |
|---------|--------|---------|-----------------|----------------|
| **Stanza** | POS/dep/lemma per language | `morphotag`, `utseg`, `transcribe` | ~200-500 MB per language | `~/stanza_resources/` |
| **Stanza (zh retok)** | Chinese tokenizer | `morphotag --retokenize --lang cmn` | ~200 MB | `~/stanza_resources/` (shared) |
| **PyCantonese** | Cantonese dictionary | `morphotag --retokenize --lang yue` | Bundled in package (~40 MB wheel) | No separate cache — installed with package |
| **Whisper** | ASR/FA models | `transcribe`, `align` | 1-15 GB depending on model size | `~/.cache/huggingface/` |
| **Wave2Vec** | Forced alignment | `align` | ~1.2 GB | `~/.cache/huggingface/` |
| **NeMo** | Speaker diarization | `transcribe` with diarization | ~500 MB | `~/.cache/nemo/` |

## Download Behavior

Models are downloaded **automatically on first use** via each library's
built-in download mechanism:

- **Stanza:** `stanza.Pipeline(download_method=DownloadMethod.REUSE_RESOURCES)`
  downloads models to `~/stanza_resources/` on first call, reuses on subsequent calls.
- **Whisper/Wave2Vec:** HuggingFace `transformers` downloads to
  `~/.cache/huggingface/hub/` with content-addressable caching.
- **PyCantonese:** Dictionary data is bundled inside the wheel — no runtime download.

### Stanza Retokenize Pipeline (Lazy Loading)

The Mandarin retokenize pipeline (`tokenize_pretokenized=False`) is **not**
loaded at worker startup. It's loaded lazily on the first `--retokenize`
request for Chinese (`cmn`/`zho`) via `load_stanza_retokenize_model()` in
`_stanza_loading.py`. The pipeline is stored under key `"{lang}:retok"` in
worker state and persists for the worker's lifetime.

This avoids loading ~200 MB of Chinese tokenizer model when retokenize is not
requested.

## Cache Locations Summary

```
~/stanza_resources/          # Stanza models (POS, dep, lemma, constituency)
~/.cache/huggingface/hub/    # Whisper, Wave2Vec, other HF models
~/.cache/nemo/               # NeMo diarization models
```

On macOS, `~/.cache/` resolves to `/Users/<username>/.cache/`. These directories
can grow to 10-30 GB with multiple languages and model sizes.

## batchalign3 Result Cache

Separately from ML model caches, batchalign3 caches NLP **results** (not models)
in a tiered cache:

| Layer | Storage | TTL | Location |
|-------|---------|-----|----------|
| **Hot** | moka in-memory | Per-process lifetime | RAM |
| **Cold** | SQLite | Persistent | `~/.local/share/batchalign3/cache.db` (Linux), `~/Library/Application Support/batchalign3/cache.db` (macOS) |

Cache keys include:
- Pipeline version (bumped on algorithm changes — see change log below)
- Language code
- MWT lexicon entries
- Retokenize flag (`|retok` suffix)

When any of these change, old cache entries are automatically ignored.

Use `--override-media-cache` to force re-computation:
```bash
batchalign3 morphotag --override-media-cache corpus/ -o output/ --lang eng
```

### Cache-Breaking Changes Log

When the morphosyntax pipeline changes in a way that produces different
results for the same input, the pipeline version constant in `cache.rs`
(`MORPHOSYNTAX_PIPELINE_VERSION`) is bumped. This invalidates all old
cached results automatically — no user action required.

| Version | Date | Change | Impact |
|---------|------|--------|--------|
| 1 | pre-2026 | Original Stanza-only pipeline | Baseline |
| 2 | 2026-03-23 | Added PyCantonese POS override for Cantonese (`yue`) | All cached Cantonese %mor results invalidated. Re-running morphotag on Cantonese files will produce corrected POS tags (佢哋→PRON instead of PROPN, etc.). Non-Cantonese cache entries unaffected but also invalidated (harmless — they will be recomputed with same results). |

**When to bump the version:**
- Adding or removing POS post-processing (e.g., PyCantonese override)
- Changing UD→CHAT mapping rules in `nlp/`
- Changing Stanza model selection for a language
- Fixing a bug that changes %mor/%gra output for existing inputs

**When NOT to bump:**
- Adding a new language (new cache keys, no collision)
- Changing cache storage implementation (keys unchanged)
- Fixing a bug in cache lookup/storage logic (not content)

## Testing with Real Models

Tests that load ML models are marked with `@pytest.mark.golden` and excluded
from the default `pytest` run. They run via:

```bash
uv run pytest -m golden    # Python golden tests
cargo nextest run --profile ml   # Rust ML golden tests
```

**Models are downloaded automatically** — the test suite does not require
manual model setup. First-run downloads may be slow (minutes for Stanza,
longer for Whisper).

**PyCantonese tests run in the default suite** because PyCantonese is bundled
(no download needed) and fast (~3s for all segmentation tests).

**OOM protection is automatic.** On machines with < 128 GB RAM, the `conftest.py`
guard forces golden tests to run sequentially (`-n 0`) even if the default
`pytest.ini` specifies parallel workers. Each Stanza model instance uses
2-5 GB — parallel workers on a 64 GB machine will OOM. The guard cannot be
bypassed; it fires per-test inside xdist workers via an autouse fixture.

### What to Expect on First Run

| Test suite | First-run download | Subsequent runs |
|------------|-------------------|-----------------|
| `uv run pytest` (default) | PyCantonese: 0s (bundled) | <1s |
| `uv run pytest -m golden` | Stanza English: ~2 min, Stanza Chinese: ~2 min | <30s |
| `cargo nextest run --profile ml` | Stanza + Whisper: ~5-10 min | <2 min |

## Cleaning Caches

```bash
# ML model caches (will re-download on next use)
rm -rf ~/stanza_resources/
rm -rf ~/.cache/huggingface/hub/
rm -rf ~/.cache/nemo/

# batchalign3 result cache (will re-compute on next run)
rm ~/.local/share/batchalign3/cache.db          # Linux
rm ~/Library/Application\ Support/batchalign3/cache.db  # macOS
```
