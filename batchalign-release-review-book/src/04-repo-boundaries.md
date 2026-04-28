# Chapter 4: Repository Boundaries and talkbank-tools

## Why This Matters

`batchalign3` is trying to become a public release surface while still leaning
on unreleased sibling code from `talkbank-tools`.

That is not sustainable release engineering.

## Current State

The clearest evidence is in the Rust workspace boundary:

- `crates/batchalign-chat-ops/Cargo.toml:22-24`
- `README.md:217-223`
- CI and release workflows cloning `../talkbank-tools`

The dependency is not optional. It is a build-time fact.

## Risks Created by the Current Boundary

### Reproducibility Risk

A tagged `batchalign3` revision does not by itself define the released system.
The effective release also depends on the state of `talkbank-tools`.

### Versioning Risk

There is no single release number that cleanly describes the whole shipped
artifact graph.

### Auditability Risk

If a researcher reports a bug, it may live:

- in `batchalign3`
- in `talkbank-tools`
- at the seam between them

Without a formal release boundary, ownership is ambiguous.

### Licensing and Compliance Risk

The main repo carries BSD-3-Clause licensing in top-level public metadata while
the Rust workspace uses `MIT` in `Cargo.toml`. Cross-repo dependency without a
formal release bill of materials makes this harder, not easier.

## What To Do

You need an explicit product relationship between the two repos. "Sibling clone"
is a developer workflow, not a release strategy.

### Recommended Options

1. Best medium-term option:
   release the required `talkbank-tools` crates properly and depend on pinned
   versions.
2. Acceptable short-term option:
   vendor the exact required crates into `batchalign3` for the public release branch.
3. Weak but better-than-now option:
   pin git SHAs explicitly and generate a release manifest that names both repos
   and revisions.

## Companion Audit Recommendation

Yes, `talkbank-tools` should be audited before public release of this stack.

The right scope is not "audit every line because it exists." The right scope is:

- release interface audit
- parser/model correctness risks that propagate into `batchalign3`
- versioning/licensing/release workflow readiness
- cross-repo compatibility guarantees

## Action Items

- Decide and document the formal release relationship between `batchalign3`
  and `talkbank-tools`.
- Remove local path dependencies from the public release path.
- Add a release manifest that records:
  - repo revision
  - dependency revision
  - version
  - license metadata
  - build inputs
- Run a companion audit of `talkbank-tools` before any joint public release.
