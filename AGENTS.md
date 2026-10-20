# Batchalign agent guidance

**Last modified:** 2026-10-20 18:07 EDT

Canonical guidance for all coding agents. Read
applicable nested guidance in the affected crates and frontend, plus the
references relevant to the task.

## First principle: typestate

Typestate and type-driven design are a firm mandate for every change, in every
language and every repository: new code, old code, fixes, refactors, tests and
test harnesses alike. Nothing below overrides this. Every project here is
long-term, never a one-off: a type is paid for once, while a bug the
architecture permits recurs for the life of the project.

- Before writing code or a test, ask which type change makes the defect
  unrepresentable, and make it. The compiler is the first test.
- Give each distinct meaning its own type and each step a transition
  `fn(Previous) -> Result<Next>` that consumes the earlier state. Put every
  check on the edge that is the only way to make the next state; consumers
  never re-check it.
- No sentinels, bare flags or catch-alls. A closed set is an enum matched
  exhaustively; a flag that chooses between behaviors is an enum; a field that
  only some states use belongs to those states.
- After each type change, count what it removed (checks, branches, panics,
  tests). If it removed nothing, the defect was moved, not eliminated.
- Every review checks for typestate opportunities explicitly. A fix that could
  have been a type is not finished.

## Scope and safety

- This repository owns the Batchalign ML/audio pipeline, Rust orchestration,
  PyO3 bridge, Python workers and dashboard. Chatter owns generic CHAT parsing,
  validation, model and transforms. Consume its published pinned dependencies;
  do not copy CHAT primitives or commit local dependency patches.
- All committed content is public. Do not add private identities, hosts, paths,
  corpus operations or incident records. Keep examples abstract and reproducible.
- Preserve the existing checkout, unrelated work and published history. Push,
  release and deploy require explicit authorization, which persists within its
  scope. Squash unpublished commits since the last push. No force-push or hook
  bypass. Routine implementation, review and verification need no repeated
  approval ceremony.
- Runtime freshness is the embedded build identity, not semver or an old install.
  Releases use the GitHub release workflow and uv-bootstrap/abi3 artifacts,
  never PyPI. Verify the artifact and live runtime for an authorized deployment.
- Do not disrupt unrelated worker jobs or silently change engine selection.
  Preserve cache provenance and compatibility checks; reuse requires matching
  inputs/options and demonstrated compatibility with the recorded producing
  implementation. A cache hit alone is not a correctness certificate. Read runtime/worker guidance before changes.
- Preserve Rust orchestration and typed worker/PyO3 contracts. Use `uv` for Python
  tooling. Regenerate affected wire/API artifacts when producers change.
  Operations over about a second must report progress through `progress_v2`.

## Design and verification

Use typestate, ownership and validated constructors to express actual invariants,
including pairing a plan with its inputs. Return typed errors in pipeline,
runner, store, FFI and background code; do not hide failures with defaults.
Chatter adjudicates CHAT validity; fix parser versus data defects at their real
source. Morphology output is UD, not legacy CLAN-mor syntax.

Retain validating-constructor, policy, wire-format and external-boundary tests.
A proof type does not justify deleting its own admission tests. Exercise the
actual failing boundary: daemon HTTP, worker engine loading, PyO3 requests, CLI
subprocesses or model transforms. Review the final diff before committing;
apply proportional checks to documentation/configuration-only edits.

Use `make help`, the current Makefile and CI workflows for supported gates.
Prefer scoped `cargo test -p <crate>`; `cargo nextest` is prohibited here.
Do not run concurrent cargo commands in one workspace. Run required final checks
on final content and reuse receipts only while inputs/tooling remain unchanged.
Shell scripts must pass default-severity shellcheck. Preserve release gates.

## Task references

| Task | Read |
| --- | --- |
| Dependencies, crate ownership and morphology | [Architecture](docs/agent-reference/architecture.md) |
| Typestate, error handling and real-boundary tests | [Design](docs/agent-reference/design.md) |
| Builds, releases and debugging | [Development](docs/agent-reference/development.md), `book/src/batchalign/developer/` |
| Pipeline/runtime/worker/cache work | `crates/batchalign/AGENTS.md` |
| Python bridge or dashboard | `crates/batchalign-pyo3/AGENTS.md`, `frontend/AGENTS.md` |
| Engine choice and provider keys (ASR, speaker diarization, `~/.batchalign.ini`) | `book/src/batchalign/user-guide/commands/diarize.md`, `book/src/batchalign/reference/command-io.md` |

This entry point resolves workflow conflicts in the references. Historical
examples are not current dependency versions or live runtime evidence. Generic
skills cannot add redundant approval pauses or authorize publishing.

<!-- trail:start -->
## Trail — repo context graph

This repo is indexed in `.trail/cache/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `trail map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `trail ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `trail grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `trail skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `trail callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `.trail/cache/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `trail ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `trail build` (deterministic,
no API key, $0).
<!-- trail:end -->
