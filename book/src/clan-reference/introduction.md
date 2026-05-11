# CLAN Command Reference

**Status:** Current
**CLAN** (Computerized Language Analysis) is a suite of commands for analyzing transcripts in [CHAT format](https://talkbank.org/0info/manuals/CHAT.html) (Codes for the Human Analysis of Transcripts). This book documents the Rust reimplementation of CLAN, invoked via the `chatter clan` command.

## Relationship to the legacy CLAN manual

This book treats the
[CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) as the primary
source for legacy command intent, examples, and option semantics when a
command is documented there. Our goal is to incorporate and improve the
non-GUI substance of that manual here, while making divergences from the
legacy C implementation explicit.

GUI-era material from the legacy manual does not belong in the CLI book. That material should instead be carried over into the documentation for the TalkBank VS Code extension, where editor workflows, inspection tools, and interactive affordances can be documented in the right place.

## What's in this book

- **Getting Started** — installation, first commands, migrating from legacy CLAN
- **User Guide** — filtering, output formats, directory workflows
- **Command Reference** — every analysis, transform, and converter command with examples
- **Architecture** — framework design, how to add commands, testing strategy
- **Developer Seams** — current CLI, validation, and dashboard boundaries to preserve while extending the system
- **Divergences** — where and why we differ from legacy CLAN

## Command overview

| Category | Count | Examples |
|----------|-------|---------|
| Analysis | 30 | FREQ, MLU, MLT, VOCD, DSS, EVAL, IPSYN |
| Transform | 18 | FLO, CHSTRING, DELIM, DATES, POSTMORTEM |
| Converter | 12 | ELAN2CHAT, PRAAT2CHAT, CHAT2SRT, SALT2CHAT |

## Why a reimplementation?

The original CLAN is a ~215,000-line C/C++ codebase maintained by a single developer. The Rust reimplementation provides:

- **Semantic AST processing** — works on parsed CHAT structure, not ad-hoc string manipulation
- **Type-safe filtering** — speaker, tier, word, gem, and ID filters via the framework
- **Multiple output formats** — text, JSON, and CSV from a single typed result
- **Golden-tested parity** — output compared against legacy CLAN binaries (95% parity, 5 accepted divergences)
- **Modern CLI** — `--flag` syntax with full backward compatibility for CLAN's `+flag` notation

When the Rust implementation differs from the legacy binary, this book tries to distinguish three cases clearly:

- semantic intent preserved, but implemented with typed AST operations instead of ad-hoc string manipulation
- deliberate modernization, such as structured JSON/CSV output or explicit errors instead of silent fallback
- unsupported legacy behavior, which should be documented as unsupported rather than imitated accidentally

## Quick example

```bash
# Word frequency for the CHI speaker
chatter clan freq --speaker CHI transcript.cha

# Mean length of utterance (JSON output)
chatter clan mlu --format json transcript.cha

# Convert ELAN annotation to CHAT
chatter clan elan2chat recording.eaf
```
