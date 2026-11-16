# Dashboard build and E2E entry points

Use Node 26 and Python 3.12 or newer. Run Make from the repository root.
`scripts/frontend_workflow.py` owns locked installation and carries admitted
artifacts through generation, verification, build and browser execution.
Strict mypy checks its phase-restricted operations.

| Command | Behavior |
| --- | --- |
| `make batchalign-dashboard-build` | Install dependencies, admit checked-in API artifacts, run Vitest and TypeScript, bundle |
| `make batchalign-dashboard-api-check` | Install dependencies, regenerate API artifacts and compare with source |
| `make batchalign-dashboard-e2e` | Install both projects, generate API artifacts, build and run mock journeys |
| `make batchalign-dashboard-e2e-real` | The same preparation plus managed server journeys |
| `bash scripts/generate_dashboard_api_types.sh` | Install dependencies and regenerate schema, types and Effect decoders |
| `bash scripts/build_react_dashboard.sh /target` | Prepare and build, then publish assets to the selected directory |

The shell adapters retain their entry points. `npm run generate:types` and
`npm run check:api` from `frontend/` delegate to the same owner. Tauri performs
generation, source comparison and bundling together. Its desktop npm project
has its own installation prerequisite.

## Runtime selection

- `BATCHALIGN_BIN` selects the schema exporter and real-server CLI. Generation
  otherwise builds the headless Rust exporter.
- `BATCHALIGN_PYTHON` selects the real server's worker interpreter.
- `BATCHALIGN_REAL_SERVER_E2E=1` enables managed server journeys.
- `BATCHALIGN_PLAYWRIGHT_WITH_DEPS=1` installs Chromium and system dependencies.
- `BATCHALIGN_SKIP_BROWSER_INSTALL=1` uses an already installed browser.
- `BATCHALIGN_SKIP_API_SYNC=1` is retained by the publication script through
  existing API artifact admission.

The ordinary bundle build consumes checked-in API artifacts because the CLI
embeds the bundle. Fresh API drift qualification separately uses the explicit
CLI or headless exporter. Generation and smoke work from a checkout with absent
frontend dependency directories.

For headed iteration after preparation,

```bash
npm --prefix frontend/e2e run test:headed
```

The canonical graph and browser responsibilities are in
`book/src/batchalign/developer/tauri-react-dashboard.md`. Browser contracts live
in `frontend/e2e/tests/` and Vitest contracts in `frontend/tests/`.
