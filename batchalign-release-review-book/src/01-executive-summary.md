# Chapter 1: Executive Summary

## Overall Assessment

The repository contains a great deal of serious engineering:

- typed Rust/Python protocol work
- a large and growing architecture book
- meaningful unit and golden coverage in several domains
- a real attempt to move CHAT semantics and orchestration out of ad hoc Python

That matters. This is not an unserious project.

But it is also not yet a release-grade product surface.

The project currently presents itself as more stable and more release-ready
than the repo evidence supports. The highest-risk gap is not "the code has no
tests" or "the team has no plan." The gap is that the project has many partial
systems that each look plausible in isolation:

- a Python wheel
- a Rust CLI
- a server/daemon
- a dashboard
- installer scripts
- a desktop shell
- a sibling dependency workspace

Yet the end-to-end public release contract tying those systems together is
still incomplete.

## What Must Be Believed for Public Release

To release this tool publicly, an outside user should be able to believe all of
the following:

1. A tagged build from tracked source produces a usable artifact.
2. That artifact has consistent version and license metadata.
3. The advertised platforms are actually supported by tested workflows.
4. The package does not depend on an unreleased sibling repository in order to
   build or function.
5. A serious bug will be caught by the test portfolio before researchers do.

Today, those statements are not defensible enough.

## Strongest Findings

The most serious findings are:

1. The wheel build is broken as a release artifact in a clean-source build.
   The package entrypoint points to `batchalign._cli:main`, which expects a
   packaged Rust binary, but the release wheel path builds only the PyO3 wheel
   and can omit that CLI binary entirely.
2. CI is not acting as a normal development gate. The main CI workflow runs on
   tag pushes and manual dispatch, not on ordinary pushes or pull requests.
3. `batchalign3` is not independently releasable while core crates still use
   local path dependencies on unreleased `talkbank-tools`.
4. The repository contains contradictory release signals:
   `1.0.0`, "First public release", "Production/Stable", and "public release
   work stays on hold" all coexist.
5. The test suite is large but strategically weak. It over-invests in type and
   serialization confidence and under-invests in packaging, integration,
   concurrency, model-loading, and platform proof.

## Recommendation in One Sentence

Do not rewrite the entire software stack from scratch. Instead, treat the next
phase as a release-stabilization program with hard stop-ship criteria around
artifact correctness, repo boundaries, test strategy, and platform support.

## Action Items

- Freeze new feature work until the stop-ship release blockers in this review
  are cleared.
- Create a named release program with owners for:
  - packaging/release engineering
  - test strategy
  - cross-platform support
  - `talkbank-tools` boundary
- Treat `talkbank-tools` as part of the release surface and audit it before any
  joint public release.
