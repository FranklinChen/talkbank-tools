# Design reference

**Last modified:** 2026-09-09 20:37 EDT

Read the sections relevant to your task. [AGENTS.md](../../AGENTS.md)
is the canonical policy entry point and resolves workflow conflicts here.
Dated incidents, measurements, versions and paths are historical evidence;
verify current source and live artifacts before relying on them. Inline
repository paths are relative to the repository root unless stated otherwise.

## Type-Oriented Design Is Mandatory

**Every change to this workspace follows type-oriented design: make illegal
states unrepresentable, and make transitions between well-defined states
explicit.** This governs new code, changes to existing code, and the design
notes that precede either. The cross-cutting rules in the next section are
instances of it, not alternatives to it.

**The principle behind the principle: an affordance beats a rule.** A
prohibition written in a guidance file loses to a type signature that makes the
forbidden thing the natural thing to write. Worked example from this workspace:
"judge staleness by build identity, never semver" was documented in two places
and still mis-applied, because the type carried a version field beside the build
id and made the wrong comparison easy. The fix was not a louder warning; it was
a type that carries a commit and no version, so the wrong comparison has no
signature to travel through.

**Shapes to recognise before writing the bug:**

- **A value proxies for a richer fact, and the two drift.** Cure: derive it from
  the fact rather than mirroring the fact.
- **A sentinel is also a legal value** (a zero that means "unset" but is also a
  real measurement; a `Default` that fabricates a fact only the caller knows).
  Cure: a variant, an `Option`, or no `Default` at all.
- **A total function silently discards information** (a parse that throws away
  the model it built; a traversal that skips what its map does not mention).
  Cure: return the information; make the lossy path the explicit one.
- **Knowledge duplicated with no owner**, held together by a contract test. Cure:
  one owner, then delete the test that existed only to detect the drift.
- **A relationship between two values is maintained by convention**, so a wrong
  pairing type-checks (a plan plus the input it was computed from; a source text
  plus an index built from a different version of it). Cure: one capability that
  owns both, so possession is the proof of the pairing.
- **A well-typed NOUN with an untyped VERB.** The type names a thing; the
  operation on it stays a loose procedure, so every caller re-derives the same
  facts, re-classifies the same errors, and restates the invariant in a comment.
  The tell is a correct-looking type surrounded by call sites that each do the
  same three things after calling it. Cure: type the OPERATION and make its
  primitives unreachable, because a public primitive is an invitation to
  re-implement the verb. See `media::Transcode`, which owns "run ffmpeg to
  produce a file" so no caller assembles an argv or cleans up a partial output.
- **A proof type built by its CONSUMERS instead of its producer.** The type is
  right, its validating constructor is honest, and it still deletes nothing,
  because each consumer builds its own while the producer hands out raw parts.
  The check moves rather than disappearing, and the count GROWS, since every
  consumer feels the need independently. The tell is arithmetic: several call
  sites validating the same relation, downstream of a producer that validates
  none of it. Cure: construct where the value is BORN. See `media::MediaWindow`,
  built by the function that computes windows rather than by its consumers.

**The self-check, and it takes ten seconds: after any type change, count what it
REMOVED** (lines, variants, constructors, checks, tests, branches). If the answer
is nothing, you did not eliminate a defect class, you relocated one, and that is
usually a net loss because the new type is a second thing to keep true. This is
the cheapest test of whether a change is type-oriented design or type-flavoured
decoration.

**Decision test, before writing any type:** name a wrong value it permits, and
ask what would notice. If the answer is a reviewer, a comment or a doc, the type
is wrong. If the answer is the compiler, it is right.

This is not a licence to rewrite. Apply it to what you touch and to new design;
existing violations get fixed when work brings you to them.

## Cross-Cutting Design Rules

1. **Types are the first layer of documentation.** Prefer named structs, enums,
   traits, and newtypes over raw primitives when a value has stable meaning.
2. **No primitive obsession at stable boundaries.** No raw strings/ints/bools for
   domain concepts (timestamps, language IDs, spans, indices, counts, engine
   selections, job/file states).
3. **No tuple-packed domain seams.** Name pairs/tuples with a struct or newtype.
4. **Avoid boolean blindness.** Use enums or state types for multiple meaningful
   states; no `tui`/`no_tui`-style bool pairs.
5. **No panic-based control flow in long-lived logic.** No `unwrap()`/`expect()`
   in pipeline, runner, store, FFI, or background paths that should report typed
   failures.
6. **Use real domain errors** (`thiserror`), not stringly failures.
7. **Keep modules browseable.** Split catch-all modules when they combine
   unrelated concerns.
8. **Use methods when they clarify ownership.** Behavior that depends on a type's
   invariants lives with that type.
9. **Touched docs need timestamps.** Any doc changed in a patch updates its
   `Last modified` field. **Always run `date '+%Y-%m-%d %H:%M %Z'`**, never guess.
10. **Time transparency.** Operations longer than ~1 second must surface to all
    UI channels (console, TUI, desktop, dashboard) via the `progress_v2` event
    channel (`batchalign/worker/_protocol.py:write_progress_event`,
    `batchalign/worker/_progress.py`). Silent waits are UX bugs. Applies to model
    downloads, model loads, external API calls, any blocking wait. Full rationale:
    [`book/src/batchalign/architecture/time-transparency.md`](../../book/src/batchalign/architecture/time-transparency.md).

## Types first, then top-down TDD for what a type cannot hold

**Before writing a failing test, ask what type change makes the defect
unrepresentable, and make that change.** The compiler is the better failing
test: red before the change and green after, at every call site, including the
callers no test enumerates, and it reports at the mistake rather than in CI. A
landing type change should DELETE the tests it obsoletes; keeping both
relocates the check instead of removing it. Writing a runtime check where a
compile error belongs looks like discipline and is a permanent tax.

**For what a type genuinely cannot hold**, and this codebase has a lot of it,
the rule below stands unchanged. Wire formats, roundtrips between two separate
functions, measurements, policy choices with real alternatives, and above all
anything reaching the outside world (a subprocess, a model on disk, an HTTP
service, another machine) are exactly what tests are for.

Then: every such feature and bug fix starts with a failing test, and the
**first** failing test is the highest-level integration test for the actual
boundary the change lives at. Unit tests on helpers are additional guards,
never substitutes.

| Bug lives at... | Top-level test invokes... |
|-----------------|---------------------------|
| BA3 daemon dispatch | HTTP POST to local `batchalign3 daemon` / `batchalign3 benchmark` |
| Worker engine selection | `load_*_engine(bootstrap)` with `monkeypatch.setattr` on the model loader |
| Rust PyO3 boundary | round-trip a real `WorkerV2Request` JSON through `execute_*_request_v2` |
| CLI argument parsing | `subprocess.run(["batchalign3", ...])` |
| CHAT transform over the model | a real CHAT fragment through `batchalign_transform::...` (generic surface comes from chatter) |

Rationale: a past release shipped multiple show-stopper defects that every
unit test passed, because none of the tests exercised the real seams
(CLI subprocess, engine loading, end-to-end pipeline). Unit-only
TDD = false green.

## Critical policy: fix root causes, never symptoms

Trace a bug to its architectural origin and fix it there. No "pragmatic"
workarounds that mask the real problem. When a bug reveals a wrong architecture,
fix the architecture.

## Rust Coding Standards

Rust **2024 edition**. Follow the project's cross-repo coding charter
(operator-maintained). High-frequency points: typed errors over panics; no
silent swallowing (`.ok()`/`.unwrap_or_default()` that hides bugs); newtypes over
primitives at boundaries; enums (with `clap::ValueEnum`) over `--flag`/`--no-flag`
pairs; `BTreeMap` for deterministic JSON in tests; `LazyLock<Regex>` for constant
patterns; files <= ~400 lines (hard limit 800); no global mutable state, inject
dependencies for test control.

