# Introduction

**Status:** Current
**Book last changed:** {{git-dates:book}}
**This page last changed:** {{git-dates:page}}

[TalkBank](https://talkbank.org/) is the world's largest open repository of spoken language data. This repository (`talkbank-tools`) is the **Batchalign3** workspace: the machine-learning pipeline that turns audio into richly annotated CHAT transcripts (automatic speech recognition, forced alignment, neural morphosyntactic tagging, and utterance segmentation), together with its web dashboard and an experimental desktop app.

The CHAT-format core (the `chatter` CLI, the Rust CHAT parsing and validation crates, the `tree-sitter-talkbank` grammar, and the CLAN command reference) now lives in the separate `chatter` project and is documented in its own book. **This book covers Batchalign3 only.**

## What Batchalign3 does

| Task | Surface | Support Status |
|---|---|---|
| Transcribe, align, or morphotag CHAT with audio and ML | `batchalign3` CLI / server | 🔷 Public preview; wheels for Windows, macOS, Linux |
| Standalone desktop GUI for Batchalign | BA3 Runner (`apps/dashboard-desktop/`) | ⚠️ Experimental only; build from source |

**Legend:** 🔷 = Public preview, ⚠️ = Experimental (not supported for end-users).

Platform and support detail live in the repo-root `docs/PLATFORM-SUPPORT.md` and `docs/RELEASE-CONTRACT.md`.

## Who this book is for

- **Researchers and clinicians** transcribing, aligning, or morphotagging audio into CHAT: start with the [Batchalign3 User Guide](batchalign/introduction.md).
- **Users coming from Batchalign2**: see the [Migration Book](batchalign/migration/index.md).
- **Contributors** to the pipeline: see the [Developer Guide](batchalign/developer/building.md).

For CHAT validation, normalization, conversion, or CLAN-style analysis without audio or ML, use the separate `chatter` project, which has its own CLI and documentation.

## Repository layout

```text
crates/         batchalign-* (runtime, types, PyO3 bridge); the CHAT-core talkbank-* crates are consumed from the chatter project
batchalign/     Python worker code (ML inference hosting)
apps/           Tauri v2 desktop app (dashboard-desktop, experimental)
frontend/       React dashboard for the Batchalign server
book/           This documentation (mdBook)
```
