# Chapter 10: Documentation, Product Surface, and Compatibility

## The Docs Are Rich but Not Fully Aligned

This repository has unusually extensive documentation for a research tool.
That is a major strength.

But documentation quality is not the same thing as documentation alignment.

## Current Contradictions

Examples:

- `batchalign/version` says first public release
- `installers/README.md` says public release remains on hold
- `pyproject.toml` says production/stable
- `book/src/developer/api-stability.md` says compatibility is still
  architecture-first and public contracts are not yet frozen

These statements cannot all be true in the same release state.

## The Product Surface Is Not Yet Narrow Enough

The project still has too many implied public surfaces:

- CLI
- server HTTP API
- dashboard
- desktop shell
- installer scripts
- Python imports
- worker wire types

Some of those are truly public. Some are transitional. Some are internal but
still look public from the outside.

For a public release, ambiguity itself is a bug.

## Compatibility Policy Needs To Become Concrete

The current internal documentation still speaks in migration terms:

- architecture-first compatibility
- aggressive redesign of cross-language boundaries
- not-frozen public contracts

That is fine during a major transition. It is not fine once researchers start
depending on the tool as a stable platform.

## Recommended Documentation Shift

The docs need a harder distinction between:

- public stable contracts
- supported but evolving surfaces
- internal implementation details
- dormant or experimental surfaces

If that distinction is not written down, contributors will keep making
reasonable but incompatible assumptions.

## Action Items

- Publish a release contract document:
  - supported commands
  - supported platforms
  - supported artifact types
  - compatibility promises
  - deprecation policy
- Remove or rewrite contradictory release-state statements.
- Label dormant/experimental surfaces explicitly.
- Keep the architecture book, but stop using it as a substitute for a public
  compatibility policy.
