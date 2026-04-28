# Tauri + React Dashboard Adoption ADR

## Date

February 25, 2026

## Status

Accepted

Release note as of 2026-03-16:
- the architecture decision stands,
- the web dashboard remains a real server-delivered surface,
- Phases 1-3 of the desktop processing flow are implemented:
  - Phase 1: command picker, native folder dialog, job submission with
    `paths_mode`, SSE-driven progress, output folder opening,
  - Phase 2: server auto-start on launch, auto-stop on exit, status bar
    with manual start/stop,
  - Phase 3: first-time setup wizard (engine selection + Rev.AI key),
    matching batchalign2's mandatory `interactive_setup()` gate,
- the CLI also gates processing commands on `~/.batchalign.ini` existence,
  matching batchalign2 behavior (was a regression, now fixed).

## Context

Batchalign3 needs one dashboard architecture that is stable over a 10-year
horizon across:

1. Browser dashboard delivery
2. Desktop operator app delivery
3. Tight API contract discipline with the Rust control plane

The prior Rust-native dashboard spike demonstrated viability, but introduced
framework-specific surface area that is no longer aligned with the preferred
long-term operating model.

## Decision

Adopt:

1. React as the canonical dashboard UI implementation.
2. Tauri as the desktop shell for the same React UI.
3. Rust OpenAPI as the canonical dashboard contract source.

## Consequences

Positive:

1. One UI codebase across web and desktop.
2. Mature ecosystem for web UI quality, testing, and observability.
3. Strong desktop packaging/update path via Tauri with minimal bespoke runtime
   code.

Negative:

1. Rust-only end-to-end UI stack is intentionally abandoned.
2. Additional Node/TypeScript toolchain ownership remains part of the platform.

## Scope Rules

1. No new product features land in deprecated dashboard prototype paths.
2. React is the dashboard feature-development target.
3. The web dashboard is the supported current public surface.
4. The desktop/Tauri shell provides the end-user processing flow (`/process`
   route) for researchers who are not comfortable with terminals. The Tauri
   side stays thin (plugins + one custom command); all UI logic lives in React.
