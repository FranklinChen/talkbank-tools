# CLI Command Surface — Proposed Restructure

**Status:** Proposed (not yet implemented)
**Last updated:** 2026-04-26 15:11 EDT

This page proposes a restructure of `batchalign3`'s top-level
command surface. It does not commit to a timeline. The current 24
top-level commands have grown organically over months of feature and
debugging work, and the result has drifted away from the
discoverability principle that should govern any CLI: the top-level
list should describe what the tool *does*, not its complete
implementation.

The 2026-04-26 incident response surfaced this concretely — a
`cancellations` debug query was about to be added as command #25,
which prompted an audit. This doc captures what that audit found and
proposes a coherent end-state. Implementation is deferred.

## Principle

**The top-level command list is the user's mental model of the
tool.** Every command in `batchalign3 --help` declares "this is a
thing you do with batchalign." Three consequences:

1. **`--help` discoverability.** A new contributor (or a successor
   five years from now per `docs/migration/phase4-succession.md`)
   first runs `batchalign3 --help`. They should see the tool's
   purpose immediately, not have to scan past 14 admin commands to
   find the 10 user-facing ones.

2. **Naming budget.** Top-level command names are scarce shared
   namespace. Spending one on a debug query (`cancellations`,
   `replay`, `bench`) is wasteful when the same command lives
   naturally one level deeper.

3. **Audience separation.** Different audiences read `--help`:
   end users, local power users, fleet operators, and people
   developing batchalign itself. Conflating them at the top level
   means everyone sees everyone else's tools.

## Current state — 24 top-level commands

```text
align  transcribe  translate  morphotag  coref  utseg  benchmark
opensmile  compare  avqi                                            ← 10 processing verbs

setup  serve  jobs  logs  cache  models  bench  doctor  replay     ← 9 admin/debug
worker(hidden)  eval  openapi  ipc-schema  version                  ← 5 misc
```

The 10 processing commands are the tool. The other 14 are scaffolding
for various audiences that have leaked into the same surface.

## Proposed restructure

Reduce visible top-level to **14 commands** (10 verbs + 4 utilities)
plus 2 hidden parents (`dev`, `worker`). Everything else moves under
purpose-specific umbrellas.

```text
batchalign3 align            ┐
batchalign3 transcribe       │
batchalign3 translate        │
batchalign3 morphotag        │
batchalign3 coref            │  10 processing verbs
batchalign3 utseg            │  (unchanged — these are what the tool does)
batchalign3 benchmark        │
batchalign3 opensmile        │
batchalign3 compare          │
batchalign3 avqi             ┘

batchalign3 setup            ← init ~/.batchalign.ini
batchalign3 version          ← print version

batchalign3 jobs LIST/SHOW/CANCELLATIONS/LOGS    ← per-job inspection
batchalign3 daemon START/STOP/STATUS/DOCTOR      ← daemon lifecycle (was `serve`)
batchalign3 cache STATS/CLEAR                    ← already an umbrella
batchalign3 eval L2-MORPHOTAG/...                ← already an umbrella

batchalign3 dev REPLAY/BENCH/OPENAPI/IPC-SCHEMA/MODELS  ← hidden from --help
batchalign3 worker START/STOP/LIST                       ← hidden, fleet only
```

Visible end-user surface drops from 24 → 14. Power-user surface
(everything visible) stays at 14 because the umbrellas absorb their
contents. Developer / dev-mode tools become discoverable via
`--help-all` or by typing the parent.

## Per-command relocation

### Stays top-level (verbs)

`align`, `transcribe`, `translate`, `morphotag`, `coref`, `utseg`,
`benchmark`, `opensmile`, `compare`, `avqi` — primary processing
commands. These ARE the tool. No discussion.

### Stays top-level (utility)

`setup`, `version` — universal utilities expected at top of any CLI.
Both are zero-friction defaults; no parent improves them.

### Becomes umbrella: `jobs`

Already user-facing, already takes a job id. Promote to an explicit
subcommand parent. Today the same `jobs <id>` magically routes to
"list jobs" vs "show job" based on whether `<id>` was provided —
that's flag-mode behaviour disguised as a positional. Explicit
subcommands replace the magic.

| Before | After |
|---|---|
| `batchalign3 jobs --server X` | `batchalign3 jobs list --server X` |
| `batchalign3 jobs <id>` | `batchalign3 jobs show <id>` |
| (none — was about to be top-level) | `batchalign3 jobs cancellations <id>` |
| `batchalign3 logs ...` | `batchalign3 jobs logs ...` (per-run logs are job-scoped) |
| (none — would benefit) | `batchalign3 jobs cancel <id>` (not yet exists) |
| (none — would benefit) | `batchalign3 jobs restart <id>` (not yet exists) |

Rationale: every command in this group operates on a specific job or
the job collection. Their natural grouping noun is "jobs."

### Becomes umbrella: `daemon` (renamed from `serve`)

`serve` is a verb form; subcommands of a verb read awkwardly
(`serve start` reads as "serve a 'start' verb"). Rename to the noun
`daemon`. The lifecycle commands then read naturally:

| Before | After |
|---|---|
| `batchalign3 serve start/stop/status` | `batchalign3 daemon start/stop/status` |
| `batchalign3 doctor` | `batchalign3 daemon doctor` (it diagnoses the daemon) |
| `batchalign3 worker` (hidden top-level) | `batchalign3 daemon worker` (hidden subcommand) |

Rationale: every command in this group manages or inspects a daemon
(local or fleet). Including `doctor` because what it actually tests
is whether the daemon's worker pipeline boots cleanly on this host.

### Stays umbrella as-is: `cache`

Already a subcommand parent (`cache stats`, `cache clear`). No
change.

### Stays umbrella as-is: `eval`

Already a subcommand parent (`eval l2-morphotag`). Tied to the
broader research agenda; researchers running an evaluation think of
it as a separate-from-processing activity. Top-level umbrella
remains correct.

### Becomes hidden umbrella: `dev`

Tools for someone developing batchalign itself, not for end users:

| Before | After |
|---|---|
| `batchalign3 replay <dump>` | `batchalign3 dev replay <dump>` |
| `batchalign3 bench` | `batchalign3 dev bench` |
| `batchalign3 openapi` | `batchalign3 dev openapi` |
| `batchalign3 ipc-schema` | `batchalign3 dev ipc-schema` |
| `batchalign3 models train/prep` | `batchalign3 dev models train/prep` |

Mark the `dev` umbrella with `#[command(hide = true)]` so it doesn't
clutter `--help`; it still works for anyone who knows to type it.
End-user help stays clean; CI / contributor scripts continue to find
their commands.

Rationale: openapi/ipc-schema export, replay, bench (perf), and
models training are all dev-mode activities. None ship as part of
"using batchalign on data." Grouping them keeps the discovery
surface honest about what's a user activity vs what's developer
infrastructure.

### Stays hidden top-level OR moves under `daemon`: `worker`

Currently a hidden top-level command for fleet daemon mgmt. Two
options:

a. **Move to `daemon worker ...`** — semantically correct (it's a
   daemon-side concern) and reduces hidden top-level count to zero.
b. **Stay top-level hidden** — fleet scripts already call
   `batchalign3 worker ...`; moving it churns pyinfra deploys.

(a) is the right end-state; (b) is the pragmatic transition. See
"Migration" below — fleet scripts should be touched at the same time
as the rename.

## What about parents that don't do anything by themselves?

`batchalign3 jobs` (no args) and `batchalign3 daemon` (no args)
should print the parent's `--help`, not error. This matches `git
remote`, `kubectl get` (when no resource type given), and `docker
context`. clap's default behaviour is correct here.

For backward-compat aliases (see below), `batchalign3 jobs` could
alias to `jobs list` for one major version cycle.

## Backward compatibility

Hard cut is hostile to existing users / scripts. Two-cycle
deprecation path:

1. **Cycle N** (the change): both old and new spellings work. Old
   spellings emit a deprecation warning to stderr:
   `warning: 'batchalign3 replay' is deprecated; use 'batchalign3 dev replay'. The old form will be removed in v2.0.`
2. **Cycle N+1**: old spellings removed; only nested forms work.

Implementation: clap's `#[command(alias = "old-name", hide = true)]`
keeps the old spelling working but invisible. Add a wrapper that
prints the deprecation warning before dispatching.

For pyinfra fleet deploys and internal scripts that call
`batchalign3 worker ...`, audit them as part of the change and
update in lock-step. The audit-then-rename is one PR; the
deprecation period covers anything we missed.

## Migration steps

When this lands (not now), the work is:

1. **Refactor `args/mod.rs`.** Reorganize the `Commands` enum into
   the new tree. Use `#[command(subcommand)]` on the new umbrellas.
   Add `alias` attributes on the old top-level forms so they keep
   working.
2. **Refactor `lib.rs::run_command`.** Replace flat dispatch with
   nested match for the umbrellas. Wrap deprecated paths in a
   `warn_deprecated_command(old, new)` helper that prints to stderr
   before dispatching to the new code path.
3. **Audit internal callers.**
   - `automation/pyinfra/deploys/` for `batchalign3 worker` usage.
   - `scripts/` in batchalign3 + workspace for any of the deprecated
     forms.
   - `book/src/` documentation for command examples.
   - `tb/src/` for shell-out usage.
   - CI workflow files for `batchalign3 openapi` / `ipc-schema`
     calls.
   - Update each to the new spelling in the same PR.
4. **Update CLAUDE.md files** in batchalign3 to reference the new
   tree.
5. **Add the deprecation-warning regression test.** Calling an old
   spelling should produce stderr containing "deprecated" and still
   succeed. Calling an unknown spelling should fail as before.
6. **Tag the cycle.** When old spellings get removed (Cycle N+1),
   commit + changelog entry calls out the breaking change and lists
   every removed alias.

## Why bother

The current surface is technically functional. Three reasons it's
worth eventually paying the migration cost:

1. **Successor cost.** The plan in
   `docs/migration/phase4-succession.md` is for an outside professor
   to inherit this codebase. Their first interaction with the tool is
   `batchalign3 --help`. The current 24-command list does not say
   "this tool transcribes / aligns / annotates linguistic data" — it
   says "this tool has 24 things, half of which are admin
   scaffolding." A new maintainer's first impression should be
   accurate.

2. **Discoverability for users.** Davida and other end users use a
   small subset of commands. Their `--help` shouldn't mix in
   `replay`, `openapi`, `worker`, `models`. Today it does.

3. **Naming budget for future work.** New debug queries
   (`cancellations` was the latest) keep proposing themselves as
   top-level. Without a structural answer for where they go, the
   pile grows. With `jobs cancellations`, `dev replay`, `daemon
   doctor` as the established pattern, future additions slot in
   cleanly.

## What this doc deliberately does NOT do

- It doesn't propose changing any *behaviour* of the commands. Every
  flag, output, and side effect stays the same. Only the path to
  invoke the command changes.
- It doesn't propose deleting any command. Nothing here is "we
  shouldn't ship X."
- It doesn't propose this for the current sprint. The user-visible
  cancellation work is unblocked by nesting `cancellations` under
  `jobs` directly when it ships; everything else can wait for a
  dedicated CLI-restructure PR.
- It doesn't propose a v2 major version bump on its own. The
  deprecation path is two cycles of any kind (point releases work).

## Cross-references

- `crates/batchalign-cli/src/args/mod.rs` — current `Commands` enum
  (24 entries).
- `crates/batchalign-cli/src/lib.rs::run_command` — current dispatch
  match.
- `book/src/architecture/cli-option-wiring.md` — how CLI flags wire
  to server options. Adjacent concern; this doc is about command
  topology, that doc is about flag plumbing inside one command.
- `book/src/user-guide/cli-reference.md` — current command catalog
  shown to users; will need updating in lock-step with any
  restructure.
- `docs/migration/phase4-succession.md` — succession plan motivating
  the discoverability principle.
