# Chapter 12: Rewrite Decision

## Should the Whole System Be Rewritten?

No.

A full ground-up rewrite of all of `batchalign3` would be the wrong move at
this point.

## Why Not

The project already has valuable hard-won assets:

- typed protocol and domain work
- substantial CHAT and algorithm code in Rust
- a clearer Python/Rust separation than before
- a large amount of documentation
- a real test base, even if misallocated

Throwing that away would reset defect discovery, reset operational knowledge,
and almost certainly extend the period of instability rather than shorten it.

## What Should Be Rewritten or Re-Architected

Several parts do justify heavy redesign:

### 1. Release Boundary

The packaging/versioning/release pipeline should be treated as redesign work,
not just bugfix work. It currently does not define a trustworthy artifact.

### 2. Repository Boundary

The `batchalign3` <-> `talkbank-tools` release relationship needs a proper
contract. That may require a significant repo-boundary redesign.

### 3. Test Strategy

The project needs a new testing architecture more than it needs more unit
tests. This is a strategic rewrite of quality gates, not of algorithms.

### 4. Selected Control-Plane Subsystems

The worker/runtime and runner/store coordination layers should be narrowed and
recomposed. This is the area where targeted subsystem rewrite can be justified.

## Decision Rule

Use targeted subsystem rewrite when all of the following are true:

1. the subsystem is large and coordination-heavy
2. failures there are operationally expensive
3. the rewrite boundary can be made explicit
4. behavior can be pinned with tests before replacement

Do not use rewrite as an escape hatch from release discipline.

## Final Recommendation

Do not rewrite the linguistic core wholesale.

Do run a hard stabilization program that is willing to:

- replace the release pipeline
- replace the cross-repo release boundary
- refactor or rewrite parts of the worker/control plane
- rebuild the testing architecture around real release risks
