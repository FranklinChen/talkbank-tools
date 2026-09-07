# Platform Support Matrix

**Status:** Current
**Last updated:** 2026-09-06

This is the canonical Batchalign platform-support matrix. Chatter owns the
CHAT core, CLI, LSP and grammar in its own repository, with independent
platform guarantees. The repository's `docs/PLATFORM-SUPPORT.md` points here
instead of maintaining a second matrix.

## CLI + Server (`batchalign3`)

| Platform | Tier | CI | Wheel | Notes |
|----------|------|----|-------|-------|
| Linux x86_64 | A | Full CI + clean-wheel CLI/server smoke | Yes | Primary CI platform |
| Linux ARM64 | B | Native release build | Yes | Built on a native ARM64 runner; no execution smoke |
| macOS ARM (Apple Silicon) | B | Clean-wheel CLI/server smoke | Yes | Release-smoke platform |
| macOS x86_64 (Intel) | B | Native release build | Yes | No execution smoke |
| Windows x86_64 | B | Clean-wheel CLI smoke | Yes | Process lifecycle uses Unix APIs (`pre_exec`, `setsid`, `killpg`); server/worker mode is not supported |

## Dashboard (React)

| Platform | Tier | Notes |
|----------|------|-------|
| All (web browser) | B | Production bundle built in CI and embedded in release wheels; browser behavior is exercised by the dashboard test suite |

## Batchalign Desktop (experimental)

| Platform | Tier | Notes |
|----------|------|-------|
| macOS / Windows | C (Experimental) | In-repo Tauri shell only; not a supported public release surface |
| Linux | C (Experimental) | No supported public desktop distribution |

This section is about **Batchalign Desktop** in `apps/dashboard-desktop/`.

## Tier Definitions

- **Tier A:** Fully CI-gated. Tests run on every PR. Regressions block merge.
- **Tier B:** Release artifacts built. Smoke-tested where possible. Not full CI coverage.
- **Tier C:** Experimental. May build, may not. No guarantees.

## Known Platform Limitations

- Worker process management uses Unix-specific syscalls (`pre_exec`, `setsid`,
  `killpg`). Windows alternatives needed for full Tier A support.
- Some contributor tooling remains shell-based; the public Windows installer
  is PowerShell and the Windows wheel receives a clean CLI smoke.
- `pyproject.toml` classifiers list macOS and Linux only (Windows build-only,
  not supported for server mode).

## Goal

Promote macOS ARM to Tier A by adding platform-specific CI test jobs.
Windows server mode requires porting Unix process lifecycle APIs before
Tier A is feasible.
