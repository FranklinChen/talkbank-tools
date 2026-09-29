---
name: ba3-dashboard-e2e
description: Run, fix, or extend the dashboard frontend's Playwright e2e tests, or diagnose a failing dashboard e2e CI job. Use for any dashboard test work.
allowed-tools: Bash, Read, Grep, Edit, Write
---

# Dashboard E2E

Canonical reference (modes matrix, scripts, CI walkthrough):
`book/src/batchalign/developer/tauri-react-dashboard.md`.

Quick entry points:
- Local real-backend run: `make batchalign-dashboard-e2e-real`
- Orchestration scripts: `scripts/run_react_dashboard_smoke.sh`,
  `scripts/build_react_dashboard.sh`,
  `scripts/check_dashboard_api_drift.sh` (API drift gate)
- CI job: `dashboard-e2e` in
  `.github/workflows/batchalign-python.yml`
- Frontend gates: `npm run build` (tsc + vite) and the vitest suite;
  generated API bindings come from `npm run generate:types` /
  `generate:schema`, never hand-edited.

Invariants (frontend/AGENTS.md): the desktop runtime seam is the one
ownership boundary; comment discipline per that file; SPA state of
record lives server-side.
