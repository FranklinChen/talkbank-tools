# batchalign3 Frontend: Dashboard + Desktop Processing UI

**Status:** Current
**Last updated:** 2026-07-25 22:45 EDT

## Overview

React SPA serving two surfaces:

1. **`/process`**: End-user processing flow for the desktop app. Researchers
   pick a command, choose files, and watch progress without a terminal.
2. **`/dashboard`**: Fleet monitoring for power users. Real-time job status,
   file-level progress, error grouping, server health, and algorithm
   visualizations.

In desktop mode (Tauri webview), `/` redirects to `/process` after any
first-launch setup gate completes.
In web mode (browser), `/` shows the dashboard.

## Tech Stack

(Versions live in `package.json`.)

| Category | Technology |
|----------|-----------|
| Framework | React |
| Language | TypeScript |
| Router | Wouter (lightweight) |
| State | Zustand |
| Data Fetching | TanStack React Query |
| Styling | Tailwind CSS |
| Build | Vite |
| API Types | openapi-typescript (auto-generated from Rust server) |
| Desktop APIs | @tauri-apps/api, @tauri-apps/plugin-dialog, @tauri-apps/plugin-shell |

## Key Commands

```bash
npm run dev              # Dev server (proxies to localhost:8000)
npm run build            # TypeScript check + Vite build
npm run generate:schema  # Regenerate OpenAPI types from Rust server
npm run check:api        # Validate API drift
npm run e2e:install      # Install Playwright deps for frontend/e2e
npm run e2e:setup        # Full e2e environment setup (install + browsers)
npm run test:e2e         # Run e2e tests against mock server (no Batchalign binary needed)
npm run test:e2e:headed  # Run e2e tests in headed mode (visible browser)
```

## E2E Testing

Canonical: `book/src/batchalign/developer/tauri-react-dashboard.md`
(modes, scripts, CI job). Quick entry: `make batchalign-dashboard-e2e-real`.

## Comment Discipline

All new and modified TypeScript files must explain their architectural role in
the code itself:

- file-level comments for modules that own routing, controller logic, state sync,
  or runtime detection
- JSDoc on exported hooks, components, and helpers
- inline comments where ownership boundaries or cache/store synchronization would
  otherwise be surprising to a new contributor

## Project Structure

See `src/` directly (a hand-maintained tree here rotted; the 2026-07
audit found it already drifting). Components live under
`src/components/`, state under `src/state/`, API bindings generated
into `src/api/`.

## Data Flow

Canonical walkthroughs: `book/src/batchalign/developer/tauri-react-dashboard.md`.

## Desktop Runtime Seam

- `runtime.ts` owns environment detection only.
- `desktop/protocol.ts` inventories raw command/event identifiers and keeps the
  transport request/response types visibly paired.
- `lib/tauri.ts` is the low-level adapter: dynamic Tauri imports, protocol
  dispatch, and browser fallbacks.
- `desktop/DesktopContext.tsx` fans that adapter out into focused React hooks:
  `useDesktopEnvironment()`, `useDesktopFiles()`, `useDesktopConfig()`, and
  `useDesktopServer()`.
- `main.tsx` must keep `DesktopProvider` above the app tree so desktop and web
  mode share one explicit runtime boundary.

## Key Patterns

- **Server-qualified keys**: `${server}|${job_id}` keeps retained aggregation
  paths collision-safe even though the released surface is single-server-first
- **WebSocket resilience**: Independent connections, exponential backoff reconnect
- **OpenAPI sync**: `npm run generate:schema` keeps TypeScript types in sync with Rust
- **React Compiler**: Babel plugin for automatic memoization
- **Desktop runtime seam**: Components/hooks consume the smallest possible
  capability hook (`useDesktopEnvironment`, `useDesktopFiles`,
  `useDesktopConfig`, `useDesktopServer`). Keep raw `@tauri-apps/*` imports and
  command/event names in `lib/tauri.ts` + `desktop/protocol.ts`, and only
  extend one capability when a new desktop-only surface is truly needed.

## Deployment

Built SPA is served by `batchalign-server` via `ServeDir` with SPA fallback
(all routes serve `index.html`, client-side routing handles the rest).

For desktop: Tauri bundles the built SPA into a native app via
`apps/dashboard-desktop/`.
