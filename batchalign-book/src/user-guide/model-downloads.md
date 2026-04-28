# Model Downloads and Caching

**Status:** Current
**Last updated:** 2026-03-23 11:35 EDT

## Automatic Model Downloads

batchalign3 downloads ML models automatically the first time you use a
command that needs them. No manual setup is required.

| Command | What Downloads | Size | First-Run Time |
|---------|---------------|------|---------------|
| `morphotag` | Stanza POS/dependency models for your language | ~200-500 MB | 1-3 minutes |
| `morphotag --retokenize --lang yue` | Nothing extra (PyCantonese is bundled) | 0 | Instant |
| `morphotag --retokenize --lang cmn` | Stanza Chinese tokenizer model | ~200 MB | 1-2 minutes |
| `transcribe` | Whisper ASR model | 1-15 GB | 5-30 minutes |
| `align` | Wave2Vec forced alignment model | ~1.2 GB | 2-5 minutes |

After the first download, models are cached locally and reused instantly.

## Where Models Are Stored

Models are cached in standard locations:

| Library | Cache Directory |
|---------|----------------|
| Stanza | `~/stanza_resources/` |
| Whisper / Wave2Vec | `~/.cache/huggingface/hub/` |
| PyCantonese | Bundled with batchalign3 (no separate cache) |

These directories may grow to 10-30 GB depending on how many languages and
model sizes you use.

## Result Caching

batchalign3 also caches NLP **results** — if you run `morphotag` on the same
file twice with the same settings, the second run reuses cached results
instead of re-running the model.

The result cache uses different keys for different settings. For example,
`morphotag` and `morphotag --retokenize` produce separate cache entries,
so switching between modes does not produce stale results.

When batchalign3 is updated with improved NLP algorithms (e.g., better POS
tagging for a language), the internal cache version is bumped automatically.
Old cached results are ignored and re-computed on next run — no user action
needed.

To force re-computation manually (e.g., after updating models):

```bash
batchalign3 morphotag --override-media-cache corpus/ -o output/ --lang eng
```

## Offline Use

Once models are downloaded, batchalign3 works fully offline. No network
access is needed for any command after the initial model download.

If you need to pre-download models for an offline environment:

```bash
# Download Stanza models for a specific language
python -c "import stanza; stanza.download('en')"
python -c "import stanza; stanza.download('zh')"
```

## Disk Space Management

To free disk space by removing cached models (they will re-download on next use):

```bash
rm -rf ~/stanza_resources/          # Stanza models
rm -rf ~/.cache/huggingface/hub/    # Whisper, Wave2Vec models
```
