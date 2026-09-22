# CLI Reference

**Status:** Current
**Last updated:** 2026-09-22 14:10 EDT

This page documents the current public `batchalign3` CLI surface. For anything
you are scripting against, confirm with `batchalign3 <command> --help`.

For detailed input/output patterns and mutation behavior per command, see
[Command I/O Parity](../reference/command-io.md).

## Command shape

```bash
batchalign3 [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS] [PATHS...]
```

Global options go before the command name.

## Global options

| Option | Meaning |
| --- | --- |
| `-v`, `-vv`, `-vvv` | Increase verbosity |
| `--workers N` | Maximum concurrent files per job (default: auto-tune; GPU commands default to 1). Auto-tune is `(ram_total_mb / 16 GB).clamp(1, 8)` for GPU-bound work. |
| `--force-cpu` | Disable MPS/CUDA and force CPU-only models |
| `--server URL` | Server to submit to. Env fallback: `BATCHALIGN_SERVER`. A loopback URL reads inputs by path; another host receives transcript text only, so commands whose inputs are recordings are refused for it. See [Server Mode](server-mode.md). |
| `--override-media-cache` | Bypass the media analysis cache (audio tasks only; text NLP tasks are not cached at all) |
| `--require-media-cache` | Require reusable evidence at cache-backed media stages; a miss fails instead of authorizing inference. Conflicts with both cache-override forms. |
| `--override-media-cache-tasks TASKS` | Bypass only named audio-evidence caches (comma-separated: `forced_alignment`, `utr_asr`, `rev_asr_evidence`, `speaker_diarization_raw_evidence`) |
| `--debug-dir PATH` | Directory for pipeline debug artifacts (CHAT/JSON fixtures for offline replay). Env fallback: `BATCHALIGN_DEBUG_DIR` |
| `--memory-tier {small,medium,large,fleet}` | Override the auto-detected memory tier (forces worker bootstrap and memory budgets for that tier regardless of actual system RAM) |
| `--timeout SECONDS` | Operator override for the audio-task transport timeout. For ASR, the default is DERIVED per request from the audio's own duration (`DecodeBudgetSeconds`, `crates/batchalign-types/src/worker_v2/requests.rs`) plus a fixed margin, not a flat number; `--timeout` can only RAISE that derived ceiling, never lower it below what the request's own decode budget needs. Forced alignment and speaker diarization still use a flat default (1800 = 30 min) unless overridden. |
| `--tui` / `--no-tui` | Toggle full-screen TUI for server-backed jobs (`DirectHost` local runs stay on terminal progress bars) |
| `--open-dashboard` / `--no-open-dashboard` | Toggle browser auto-open for submitted server job pages (macOS only, interactive TTY only) |
| `--engine-overrides JSON` | Per-engine PARAMETERS, as a `{string:string}` JSON object, e.g. `{"qwen_model":"Qwen/Qwen3-ASR-0.6B-hf","qwen_device":"cpu"}`. Forwarded to the worker as opaque knobs. The `asr` / `fa` / `utr` / `translate` keys additionally select an engine and beat the per-command flags; see "Engine selection" below. The payload is parsed once, while the command line is parsed, so an invalid one is rejected before anything runs. A bare engine name (`--engine-overrides whisper`) is reported as the wrong-flag mistake it is, naming `--asr-engine` and its siblings, rather than as malformed JSON. |
| `--sequential` | Process files one at a time with a single worker. No memory gate, no server. Ideal for small jobs on laptops |
| `--no-server` | Skip auto-detection of a local server; force direct in-process execution |

BA2 compatibility flags (`--memlog`, `--mem-guard`, `--adaptive-workers`,
`--pool`, `--shared-models`, etc.) have been removed. If your scripts use them,
remove them.

## Engine selection

**These are per-command flags, not global ones**: they go AFTER the command
name, like `batchalign3 transcribe in/ --asr-engine paraformer`.

| Option | Commands | Meaning |
| --- | --- | --- |
| `--asr-engine NAME` | `transcribe`, `benchmark` | ASR engine. |
| `--fa-engine NAME` | `align` | Forced-alignment engine. |
| `--utr-engine NAME` | `align` | Utterance-timing-recovery engine. Only consulted with `--utr`. |
| `--existing-wor-boundaries {preserve,rebuild-from-evidence}` | `align` | v0.4.0 option for prior `%wor`/main boundaries. Default `preserve`; rebuild mode is experimental and does not change raw FA cache identity. |
| `--end-overlap-policy {clamp-all-adjacent,preserve-cross-speaker}` | `align` | Same-speaker (default, `preserve-cross-speaker`) or every adjacent pair (`clamp-all-adjacent`) end overlap resolved from measured word hulls. Cross-speaker overlap is left alone under the default, since it is ordinary conversation. No raw FA cache-key change. |
| `--translate-engine NAME` | `translate` | Translation engine. |

Each flag's `--help` lists every value it accepts, derived from the engine
enum itself, so the advertised set and the accepted set are the same list. An
unrecognized name is rejected while parsing, naming what you typed and
suggesting the nearest valid value.

The older `--asr-engine-custom` / `--fa-engine-custom` / `--utr-engine-custom`
flags still work and are hidden from help. They exist because each visible flag
used to advertise only some of its engines, which left the rest reachable only
through a second, differently-named flag: that is how the Cantonese engines
stayed hidden from the people who needed them. Prefer the flags above.

The `asr` / `fa` / `utr` / `translate` keys of `--engine-overrides` also select
an engine, and take precedence over the flags. Use the flags for ordinary work;
the keys exist so one shared option can pin engines across a batch.

`utr` was missing from that set until 2026-08-06: it parsed, was forwarded to
the Python worker as an opaque extra, and was ignored by everything, so a user
who wrote it saw their choice silently dropped.

## Sequential mode

`--sequential` gives you the simplest possible execution path, similar to
batchalign2's direct mode. One worker per task type, files processed one at a
time, no concurrency infrastructure:

```bash
batchalign3 morphotag corpus/ -o output/ --sequential
```

**What it does:**
- Forces `--workers 1` and `--no-server`
- Disables the memory gate (no cross-process coordination)
- Keeps the worker alive for the entire run (no idle timeout kills)
- Preserves the utterance cache (repeated runs benefit from cached results)

**When to use it:**
- Processing a handful of files on a laptop
- Debugging pipeline issues (predictable, single-threaded execution)
- Environments where memory auto-tuning is unwanted

**When NOT to use it:**
- Large corpus runs (50+ files), the default parallel mode is 3-5× faster
- Fleet machines with warm workers, use the server instead

`--sequential` is incompatible with `--server` (mutually exclusive).

## Dashboard browser auto-open

On macOS, when you run a processing command interactively (e.g.,
`batchalign3 transcribe corpus/ output/`), the CLI automatically opens the
job's dashboard page in your default browser. This lets you monitor progress
in real time.

Direct local execution does not submit an HTTP job, so there is no dashboard
page to open. In that mode, `--open-dashboard` is a no-op and the CLI shows
local terminal progress inline instead.

The dashboard auto-open is **only** triggered when:

- Running on macOS (no-op on Linux/Windows)
- stderr is connected to an interactive terminal (TTY)
- `--no-open-dashboard` was not passed
- The `BATCHALIGN_NO_BROWSER` environment variable is not set

It will **not** fire in non-interactive contexts: cron jobs, CI pipelines,
SSH sessions without a display, piped output, or scripts. To suppress it
explicitly in interactive sessions, pass `--no-open-dashboard`.

## Common path-processing options

The core processing commands documented below all accept:

| Option | Meaning |
| --- | --- |
| `PATHS...` | Input files or directories |
| `-o`, `--output DIR` | Output directory |
| `--file-list FILE` | Read input paths from a text file (see below) |
| `--in-place` | Modify inputs in place |

When exactly two positional paths are provided, the CLI still accepts the
legacy input/output directory form. For new scripts, prefer `-o/--output`.

### `--file-list` format

`--file-list FILE` reads input paths from a plain-text UTF-8 file, one path
per line:

- Blank lines and lines beginning with `#` are ignored; whitespace around
  each path is trimmed.
- A **relative path resolves against the directory containing the list
  file**, not the directory you run the command from. Absolute paths are
  used as written.
- A path naming a **directory** is expanded exactly like a positional
  directory argument: matching files are discovered recursively beneath it.
- The input set is **de-duplicated, keeping the first occurrence**. A file
  listed twice, spelled two ways (`a.cha`, `./a.cha`, an absolute path), or
  reached both directly and through a listed directory is processed once.
- Every entry must exist when the command runs. A missing entry is a usage
  error (exit code 2) naming the list file and line, for example
  `lists/rerun.txt:3: input path does not exist: lists/corpus/missing.cha`.

```text
# lists/rerun.txt: entries are relative to lists/
corpus/session-01.cha
corpus/session-02.cha

# a whole directory, and an absolute path
corpus/follow-up/
/data/project/extra/session-09.cha
```

```bash
# Run align on every listed input (in place, against a remote server)
batchalign3 --server http://your-server:8001 align --file-list lists/rerun.txt
```

To process a large list in smaller batches, split it into chunk files in
the same directory as the original list (for example, run
`split -l 10 rerun.txt batch-` inside that directory) so relative entries
keep resolving against the same place, then run
`batchalign3 align --file-list <chunk>` on each chunk sequentially.

`--file-list` cannot be combined with positional `PATHS` arguments; the CLI
rejects the combination. Without `-o/--output`, every listed input is
processed in place (output overwrites input); with `-o DIR`, results are
written under `DIR` exactly as for positional inputs.

For batched text-NLP commands (`morphotag`, `utseg`, `translate`, `coref`),
large `--file-list` runs may not show file-by-file on-disk rewrites while the
invocation is still running. The command can batch/stage work internally and
then commit the in-place writes when the current invocation finishes. If you
need visible write-through during a long rerun, split the list into smaller
chunks and run those chunks sequentially.

## Processing commands

Each processing command has a dedicated page with full options, a pipeline
diagram, examples, and gotchas. Click the command name for complete
documentation.

### CHAT-mutation commands (input `.cha` → output `.cha`)

| Command | What it does |
| --- | --- |
| [**align**](commands/align.md) | Add word-level and utterance-level timestamps via forced alignment |
| [**morphotag**](commands/morphotag.md) | Add `%mor` POS/lemma and `%gra` dependency tiers |
| [**utseg**](commands/utseg.md) | Re-segment utterance boundaries using Stanza constituency parsing |
| [**translate**](commands/translate.md) | Add `%xtra` English translation tiers |
| [**coref**](commands/coref.md) | Add sparse `%xcoref` coreference annotation tiers (English only) |
| [**compare**](commands/compare.md) | Compare against gold `.cha` references; write `%xsrep`/`%xsmor` + `.compare.csv` |

### Audio-input commands (input audio → new files)

| Command | What it does |
| --- | --- |
| [**transcribe**](commands/transcribe.md) | Create `.cha` transcripts from audio via ASR |
| [**benchmark**](commands/benchmark.md) | Transcribe and evaluate WER against gold `.cha` references |
| [**opensmile**](commands/opensmile.md) | Extract acoustic features → `.opensmile.csv` (positional I/O) |
| [**avqi**](commands/avqi.md) | Calculate Acoustic Voice Quality Index from paired `.cs`/`.sv` audio (positional I/O) |

## Operational commands

### `setup`

Initialize `~/.batchalign.ini`:

```bash
batchalign3 setup
batchalign3 setup --non-interactive --engine whisper
batchalign3 setup --non-interactive --engine rev --rev-key <KEY>
```

Options:

| Option | Meaning |
| --- | --- |
| `--engine {rev,whisper}` | Persist default ASR engine |
| `--rev-key KEY` | Rev.AI key for non-interactive setup |
| `--non-interactive` | Disable prompts |

### `logs`

```bash
batchalign3 logs
batchalign3 logs --last
batchalign3 logs --export
batchalign3 logs --clear
```

Key options:

| Option | Meaning |
| --- | --- |
| `--last` | Show the most recent run log |
| `--raw` | Raw JSONL output with `--last` |
| `--export` | Zip recent logs |
| `--clear` | Delete log files |
| `--follow` | Tail the newest log file |
| `-n`, `--count N` | Number of recent runs to list |

### `serve`

```bash
batchalign3 serve start --foreground
batchalign3 serve status
batchalign3 serve stop
```

`serve start` key options:

| Option | Meaning |
| --- | --- |
| `--port PORT` | Listen port |
| `--host HOST` | Bind address |
| `--config PATH` | Alternate `server.yaml` path |
| `--python PATH` | Worker Python executable |
| `--foreground` | Do not daemonize |
| `--test-echo` | Start test-echo workers |

### `jobs`

```bash
batchalign3 jobs --server http://myserver:8000
batchalign3 jobs --server http://myserver:8000 <JOB_ID>
batchalign3 jobs <JOB_ID>
batchalign3 jobs --json <JOB_ID>
batchalign3 jobs cancellations <JOB_ID>
```

With `--server`, lists or inspects remote jobs. Without `--server`,
inspects the local job artifact directory for post-failure debugging.
Pass `--json` for machine-readable output.

The `cancellations` subcommand prints the cancellation audit history
for a single job, every cancel attempt is recorded with `source`
(tui / api / dashboard / staging / signal), `host`, `pid`, `reason`,
and `in_flight_filename`. Use this when a user reports "I didn't
cancel that job."

### `cache`

```bash
batchalign3 cache stats
batchalign3 cache clear --yes
batchalign3 cache clear --all --yes
```

`BATCHALIGN_ANALYSIS_CACHE_DIR` and `BATCHALIGN_MEDIA_CACHE_DIR` relocate
the underlying caches for isolated runs. BA2-compatible flag forms
`cache --stats` and `cache --clear` are still accepted.

### `openapi`

```bash
batchalign3 openapi -o openapi.json
batchalign3 openapi --check --output openapi.json
```

`--check` exits non-zero when the target file does not match the generated
schema.

### `models`

Two subcommands:

| Subcommand | Purpose |
| --- | --- |
| `models prep` | Extract training text from CHAT files (Rust-native, no CLAN needed) |
| `models train` | Forward to the Python training runtime (`python -m batchalign.models.training.run`) |

See [Models Training Runtime ADR](../decisions/models-training-runtime-adr.md).

### `ipc-schema`

```bash
batchalign3 ipc-schema -o schemas/
batchalign3 ipc-schema --check --output schemas/
```

Emits JSON Schema for Rust→Python IPC types. Without `-o`, schemas are
written to stdout as a single JSON object. With `--check`, exits non-zero
on schema drift against the target directory.

### `bench`

```bash
batchalign3 bench <COMMAND> <IN_DIR> <OUT_DIR> [--runs N]
batchalign3 bench align corpus/ out/ --runs 3 --dataset eng-childes-v1
batchalign3 bench align corpus/ out/ --use-cache
```

Benchmark command execution time across repeated runs. `<COMMAND>` is
one of: `align`, `transcribe`, `transcribe_s` (with diarization),
`morphotag`, `translate`, `utseg`, `benchmark`, `opensmile`, `coref`,
`compare`. Distinct from the `benchmark` top-level command, which
measures ASR word accuracy.

| Option | Meaning |
| --- | --- |
| `--runs N` | Number of repeat runs (default: 1) |
| `--dataset LABEL` | Dataset label included in structured output (useful for cross-run comparison) |
| `--use-cache` | Use the analysis cache for benchmark runs (default: bypass cache so each run hits cold paths) |

### `doctor`

```bash
batchalign3 doctor
batchalign3 doctor --lang yue --format json
batchalign3 doctor --explain memory_gate_mb
batchalign3 doctor --warnings-as-errors
```

Pre-flight diagnostic that spawns a test worker, sends known inputs
through the morphosyntax pipeline, and validates the output structure.
Catches machine-specific issues (stale models, missing processors, MWT
quirks) before they become production failures.

| Option | Meaning |
| --- | --- |
| `--lang LANG` | Language to test (default: `eng`) |
| `--format {human,json}` | Output format (default: `human`) |
| `--python PATH` | Custom Python path (overrides `BATCHALIGN_PYTHON`) |
| `--explain KNOB` | Trace why one resolved knob has its current value (`gpu_thread_pool_size`, `force_cpu`, `max_total_workers`, `max_concurrent_jobs`, `max_workers_per_key`, `memory_gate_mb`). Prints resolved value, source (operator override vs. host-facts recommendation), the rule that produced the recommendation, and the relevant detected facts. Implies `--check`. |
| `--warnings-as-errors` | Treat host-facts validation warnings as fatal: exit non-zero when any warning fires, not only on error. Intended for CI gates that want zero-warning deployments. Has no effect outside `--check` / `--explain`. |

### `replay`

```bash
batchalign3 replay <DUMP_FILE>
batchalign3 replay --lang yue path/to/failed_ipc_*.json
```

Replay a captured failed IPC request against a fresh worker. Takes a
dump file from `~/.batchalign3/debug/` and sends the exact request to
a new worker, reporting the response. Useful for reproducing field
failures locally.

### `eval`

```bash
batchalign3 eval l2-morphotag <ARGS>
batchalign3 eval utr-alignment --chat <CHAT> --tokens <JSON> --output <JSON>
batchalign3 eval utseg-replay post-chat --input-chat <CHAT> --evidence <JSON> --output-chat <CHAT>
batchalign3 eval utseg-replay pre-asr --asr-response <JSON> --evidence <JSON> --output-chat <CHAT> [--media-name <NAME>] [--wor]
```

Evaluation subcommands. Currently:

| Subcommand | Purpose |
| --- | --- |
| `eval l2-morphotag` | L2 morphotag evaluation: pair `@s` words with `%mor` / `%gra` items via typed AST walk (supersedes `scripts/l2-eval/analyze.py`) |
| `eval utr-alignment` | Offline global UTR word-to-token replay with fingerprinted typed evidence and no inference or CHAT mutation |
| `eval utseg-replay` | Reapply retained utterance-boundary evidence and report whether it still reproduces the document the run wrote; exits 1 on a difference, 2 on a refused input |

### `version`

```bash
batchalign3 version
```

Prints version and build information.

## Exit codes

`batchalign3` uses stable non-zero exit code categories:

| Code | Meaning |
| --- | --- |
| `2` | Usage/input error |
| `3` | Configuration error |
| `4` | Network/connectivity error |
| `5` | Server/job lifecycle error |
| `6` | Local runtime error |

Exit code `1` is reserved for unexpected failures outside the typed categories.

A server that reports another build than the CLI's, or no build, is refused
before anything is submitted, with exit code `5`; see
[Server Mode: build identity check](server-mode.md#build-identity-check).
