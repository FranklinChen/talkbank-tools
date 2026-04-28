# Batchalign2 -> Batchalign3 Migration Book

**Status:** Current
**Last updated:** 2026-03-28 17:54 EDT

## Scope

This migration book explains the transition from **batchalign2 baseline commit**
`84ad500b09e52a82aca982c41a8ccd46b01f4f2c` (2026-01-09) to the current
**batchalign3** architecture.

Secondary comparison point when needed:

- released `batchalign2` master-branch point:
  `e8f8bfada6170aa0558a638e5b73bf2c3675fe6d` (2026-02-09)

Audience:

- users migrating command-line workflows,
- developers/contributors migrating implementation work.

## Why this migration is not a patch release

Batchalign3 is not "batchalign2 plus fixes." It is a structural rewrite of the
format/runtime core:

- CHAT parsing/validation/serialization moved from ad-hoc Python text logic to
  Rust AST operations.
- job orchestration expanded from local dispatch to daemon + server job modes.
- an intermediate plugin phase was retired; the current release uses in-tree
  engines plus `batchalign.pipeline_api`, not a public entry-point plugin loader.
- avoidable runtime dynamic-programming remap paths were narrowed or removed in
  favor of deterministic identity/index/interval mapping.

The migration also includes durable user-visible improvements that matter to
existing BA2 users:

- higher correctness for `%mor`/`%gra`, retokenization, and timing writeback,
- faster repeat runs from explicit daemon/server execution and first-class
  utterance-level caching,
- clearer operational surfaces for long jobs (`serve`, `jobs`, `logs`; `openapi`
  is contributor-facing),
- stricter data-structure boundaries so token/word identity is preserved
  instead of reconstructed after flattening.

The key engineering theme across commands is the elimination of string-based
pipelines that produced silently wrong output:

- no more ad-hoc string surgery on CHAT text,
- no more parallel-array patching that drifted when tokenizers disagreed,
- no more broad "just run DP on the flattened text" recovery that masked
  upstream errors with plausible-looking guesses,
- replaced by stable word identity, explicit indexing, structured AST
  iteration, and typed validation throughout.

This page is the summary layer. Detailed command-by-command changes now live in:

- [User Workflow Migration](user-migration.md)
- [Developer Architecture Migration](developer-migration.md)
- [Algorithms, Language, and Alignment Migration](algorithms-and-language.md)

## Quick delta map

| Area | batchalign2 @ 84ad500 | batchalign3 |
|---|---|---|
| Core CHAT handling | Python lexer/parser/generator + string transforms | Rust parser + typed AST + serializer + structured validation |
| Alignment remap strategy | DP-heavy fallback remapping and post-hoc reconstruction | Identity/index-first deterministic mapping with narrower, explicit fallback policies |
| Runtime topology | Primarily local CLI dispatch | Local daemon, HTTP server, jobs/logs tooling, contributor-facing OpenAPI export |
| Concurrency | Sequential file processing (Jan 9); concurrent dispatch added in Feb 9 but job-scoped | Daemon/server job lifecycle, persistent worker subprocesses, resumable state |
| Extensibility | Forking/custom branches | In-tree engines plus `batchalign.pipeline_api`; no public entry-point plugin loader in the current release |
| UI/ops | Terminal-centric | Web dashboard plus health/jobs/log surfaces; desktop/Tauri launcher deferred from first release |
| Test posture | Lower coverage and fewer corpus gates | broad golden/integration suites + policy guards |

## Comparison states and policy

This book now works from a **dual-baseline** policy:

| State | What it represents |
|---|---|
| Jan 9, 2026 `batchalign2-master` `84ad500...` | the primary migration anchor for core / non-HK behavior |
| Jan 9, 2026 `~/BatchalignHK` `84ad500...` | the primary migration anchor for HK / Cantonese behavior |
| Feb 9, 2026 `batchalign2-master` `e8f8bfa...` | the later released BA2 master-branch tracking point |
| current `batchalign3` | the present Rust-first control plane and worker architecture |

The primary comparison is Jan 9 anchor → current BA3:

- use the Jan 9 `batchalign2-master` anchor for core / non-HK migration claims
- use the Jan 9 `BatchalignHK` anchor for HK / Cantonese migration claims
- use the Feb 9 BA2 point only when you specifically need the later released
  BA2 master-branch surface as secondary context

Transient unreleased intermediate states are not cataloged here.

## Comparison discipline for release work

The canonical migration baseline is always one of the Jan 9 anchors above.

- Use Feb 9 BA2 only as secondary context when you specifically need the last
  released BA2 master-branch behavior.
- Do **not** use later Python operational packaging, fleet wheels, or other
  deployment artifacts as the migration baseline. Those are useful deployment
  references, but they blur the actual Jan 9 anchor → BA3 migration delta.
- For HK material, remember that the historical baseline command is
  `batchalignhk`, not stock `batchalign`.
- For preserved Jan 9 runners, keep the native legacy CLI shape:
  `command inputfolder outputfolder`.

For concrete local comparison work, this repo now carries two complementary
tools:

- `scripts/stock_batchalign_harness.py` for curated `benchmark` cases with an
  explicit allowlist of known Jan 9 BA2 baseline bugs
- `scripts/compare_stock_batchalign.py` for raw side-by-side output diffing
- [`feature-parity-audit.md`](feature-parity-audit.md) as the running record of
  live curated results

In both cases, point the baseline executable at a runner explicitly pinned to
the correct Jan 9 anchor:

- core / non-HK: a pinned `batchalign` runner for `batchalign2-master`
- HK / Cantonese: a pinned `batchalignhk` runner for `BatchalignHK`

## How to read this migration book

1. Start with [User Workflow Migration](user-migration.md) for command/runtime
   behavior and release-surface deltas.
2. Then read [Developer Architecture Migration](developer-migration.md) for
   control-plane, typed-contract, and codebase-structure changes.
3. For algorithmic behavior (alignment, retokenization, multilingual/Japanese,
   DP), read [Algorithms, Language, and Alignment Migration](algorithms-and-language.md).
4. For engine-extension details, read [HK/Cantonese Engines: Migration and Architecture](../architecture/hk-cantonese-engines.md) and [Adding New Engines](../developer/adding-engines.md).

## Relationship to existing detailed references

This book is the migration crosswalk. Deep subsystem specifics remain in the
existing architecture/reference chapters (CHAT parsing, forced alignment,
multilingual, MWT, Japanese morphosyntax, HK engine architecture, server
architecture, dynamic-programming policy).

## "Every change" interpretation and audit method

The migration scope is broad enough that "every change" is covered by subsystem
catalog, not by line-by-line patch replay. This book therefore provides:

- explicit baseline anchoring (`84ad500...`),
- optional later-BA2 release anchor (`e8f8bfa...`) when needed for the last
  shipped master-branch behavior,
- command/runtime/architecture/algorithm/engine-extension change catalogs,
- pointers to subsystem references where each class of change is fully specified,
- practical user and contributor migration checklists.

Intermediate migration campaign notes (for example branch-by-branch progress
logs and implementation spikes) are not treated as canonical book content. The
book keeps only current-state behavior plus baseline migration crosswalks.
