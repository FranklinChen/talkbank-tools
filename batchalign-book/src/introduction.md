# Introduction

**Batchalign** is a toolkit for language sample analysis (LSA) from
the [TalkBank](https://talkbank.org/) project.  It processes conversation
audio files and their transcripts in CHAT format, providing automatic speech
recognition, forced alignment, morphosyntactic analysis, translation,
utterance segmentation, and audio feature extraction.  Batchalign runs on
**Windows, macOS, and Linux**.

The standalone `batchalign3` binary (written in Rust) provides the CLI and
an HTTP server for offloading work to a central machine.  Python ML workers
(Stanza, Whisper, etc.) handle inference and are managed automatically by
the server.

## Who is Batchalign for?

Batchalign is designed for researchers and clinicians who work with
conversation transcripts -- particularly those stored in TalkBank's CHAT
format.  Typical workflows include:

- **Transcribing** recorded conversations into CHAT files via ASR (Rev.AI or
  OpenAI Whisper).
- **Aligning** existing transcripts against audio to produce word-level and
  utterance-level timestamps.
- **Tagging** transcripts with morphological and dependency analyses (`%mor`
  and `%gra` tiers) using Stanford Stanza.
- **Translating** non-English transcripts to English.
- **Segmenting** unsegmented text into utterances.
- **Extracting** acoustic features (OpenSMILE, AVQI) from speech recordings.

## Key features

- **Rust-backed CHAT parsing.**  All CHAT reading and writing goes through a
  Rust AST (`batchalign_core`), ensuring correct handling of CHAT's complex
  encoding, escaping, and continuation rules.
- **Per-utterance caching.**  Morphosyntax, forced alignment, and utterance
  segmentation results are cached in a local SQLite database so that
  reprocessing the same corpus is nearly instant.
- **Server mode.**  A built-in HTTP server lets you offload processing to a
  central lab machine. Clients send small CHAT files (~2 KB each); the server
  resolves media from configured volume mounts and does all the heavy
  computation.
- **Automatic concurrency tuning.**  The CLI auto-tunes worker counts based
  on available RAM and GPU resources, and manages a persistent local daemon so
  model loads are amortized across successive commands.
- **Python API.**  In addition to the command-line interface, Batchalign
  exposes a Python library for building custom analysis pipelines.

## How to use this book

This book is organized into six sections:

1. **Migration Book** -- the authoritative public crosswalk from the previous
   release to the current version, anchored to the January 9, 2026 baseline
   `84ad500...` and, where needed, the February 9, 2026 released BA2 master
   point `e8f8bfa...`.
2. **User Guide** -- Installation, quick start, CLI reference, Python API,
   server setup, and troubleshooting.
3. **Architecture** -- How the pipeline, engine, dispatch, caching, and
   validation systems work internally.
4. **Technical Reference** -- Detailed documentation on CHAT format,
   morphosyntax, forced alignment, multilingual support, and more.
5. **Developer Guide** -- Building from source, testing conventions, adding
   new engines, and working with the Rust core.
6. **Design Decisions** -- ADRs and accepted design notes on the implemented Rust
   control plane, correctness work, and server orchestration.

If you are a new user, start with [Installation](user-guide/installation.md)
and [Quick Start](user-guide/quick-start.md). If you are migrating from
a previous version, start with [Migration Guide](migration/index.md). If
you want to use Batchalign as a Python library, see
[Python API](user-guide/python-api.md).

## Acknowledgments

The TalkBank Project, of which Batchalign is a part, is supported by NIH
grant HD082736.

If you have questions or encounter issues, please open an issue in the
repository's issue tracker.
