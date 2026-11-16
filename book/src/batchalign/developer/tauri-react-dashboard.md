# Tauri + React Dashboard

**Status:** Current
**Last updated:** 2026-11-15 21:03 EST

## Overview

Batchalign3 dashboard delivery uses:

1. `frontend/` as the canonical React UI.
2. `apps/dashboard-desktop/` as the Tauri desktop shell.

The frontend serves two surfaces:

- **`/process`**: End-user processing flow. Researchers pick a command, choose
  files via native folder picker, and watch observed job progress. Default landing
  page in desktop mode.
- **`/dashboard`**: Fleet monitoring for power users. Real-time job status,
  file progress, error grouping, algorithm visualizations.

The Tauri shell stays thin: a small set of custom commands for file discovery,
config I/O, output-directory opening and managed-service discovery, plus the
`dialog` plugin for file pickers. All shared UI logic lives in the React
frontend, which now consumes shell-only capabilities through an explicit
desktop runtime seam instead of scattering raw Tauri imports across hooks and
components.

The following diagram shows the component hierarchy across both surfaces
and the Tauri backend IPC boundary.

```mermaid
flowchart TD
    subgraph "React Frontend (frontend/src/)"
        subgraph "/dashboard: Fleet Monitoring"
            dash["DashboardPage"]
            dash --> joblist["JobList"] & workers["WorkerProfilePanel"] & mem["MemoryPanel"] & vitals["VitalsRow"]
            joblist --> jobcard["JobCard\n(status badge, progress)"]

            detail["JobDetailPageView"]
            detail --> filetable["FileTable\n(directory-grouped rows)"] & errors["ErrorPanel\n(error groups by code)"] & stage["PipelineStageBar\n(5 phases)"]
        end

        subgraph "/process: End-User Processing"
            process["ProcessPage"]
            process --> cmdpicker["CommandPicker\n(2x3 grid)"]
            process --> folder["FolderPicker\n(native dialog)"]
            process --> form["ProcessForm"]
            process --> progress["ProcessingProgress\n(job-record observations)"]
        end
    end

    subgraph "Tauri Backend (src-tauri/src/)"
        discover["discover_files()"]
        start["start_server()"]
        reveal["open_output_directory()"]
        status["server_status()"]
        config["read_config() / write_config()"]
    end

    subgraph "React Hooks (frontend/src/desktop/)"
        lifecycle["useServerLifecycle"]
        health["useServerHealth"]
        submit["useSubmitJob"]
        stream["useJobStream\n(GET job record)"]
    end

    lifecycle -->|"Tauri IPC"| status
    folder -->|"Tauri IPC"| discover
    form -->|"HTTP POST /jobs"| submit
    progress -->|"GET /jobs/id"| stream
    progress -->|"Tauri IPC"| reveal
```

## Compiler, generated contracts and build configuration

TypeScript 7 checks application code, contract fixtures, Vite configuration,
Vitest configuration and the decoder generator. `npm run typecheck` checks both
`tsconfig.json` and `tsconfig.tools.json`. `npm run build` runs those checks
before bundling. Unsupported plugin options therefore fail at compilation.

Vite 8's React plugin uses Oxc for JSX and refresh. React Compiler runs through
`@rolldown/plugin-babel` with `reactCompilerPreset()` and Babel 8. The preset
filters files eligible for React compilation. Verify the integration against
production transformed output, where compiled components import the React
compiler runtime and access their memoization caches.

`openapi-typesafe` generates `src/generated/api.ts`. The checked decoder script
compiles the same Rust-produced OpenAPI 3.1 document into Effect schemas in
`src/generated/decoders.ts`. Both generated artifacts belong to API drift
checking. The importer refuses unsupported schema constraints. Descriptions,
titles and examples stay in the OpenAPI document, while the browser carries
constraints and reference identities. Decoders retain uninspected JSON
extensions while UI types expose the declared fields.

Regenerate through `scripts/generate_dashboard_api_types.sh` or
`npm run generate:types`. Exact versions of the release-candidate generator
and Effect are recorded in package.json and the lockfile. Effect's socket
module and Vitest adapter have evolving APIs, so upgrades qualify the complete
transport and test graph together.

```mermaid
flowchart LR
    rust["Rust API types"] --> openapi["OpenAPI document"]
    openapi --> types["openapi-typesafe types"]
    openapi --> schemas["Effect runtime decoders"]
    http["HTTP response bytes"] --> schemas
    schemas --> record["Admitted job and file observations"]
    record --> progress["ObservedProgress and retained file index"]
    frame["WebSocket frame and prepared server"] --> envelope["Envelope and payload admission"]
    schemas --> envelope
    envelope --> update["DashboardUpdate with server provenance"]
    update --> cache["Query and Zustand reconciliation"]
```

## Dependency and artifact preparation

`scripts/frontend_workflow.py` owns locked dependency admission, API generation,
source comparison, bundling and browser preparation. Shell, Make and Tauri build
entry points delegate to this owner. Bootstrap uses Python 3.12 or newer and
Node 26. Node-only CI jobs explicitly select their Python interpreter.

`WorkspaceLease.acquire` holds the checkout's build lock through the complete
invocation, serializing dependency and generated-asset writers. The admitted
lease is carried through each project's dependency state. `InstalledNpm.prepare`
runs `npm ci` and checks both manifests against their
captured identities. Its private constructor binds the resolved npm executable,
directory and manifests to that successful installation. Frontend and E2E
projects each receive an admitted dependency state.

The frozen, final `Frontend[Stage]` restricts operations through explicit self
types. Strict mypy checks the graph in the development-workflow gate. Fresh
generation retains the Rust schema and emitted TypeScript paths. Source
comparison retains its completed result in `Frontend[Verified]`. Building
admits `dist/index.html` and produces `Frontend[Built]`, which exposes smoke
execution. Public constructors require an uninhabited argument.

```mermaid
flowchart TD
    source["Frontend package and lockfile"] --> lease["WorkspaceLease<br/>exclusive checkout ownership"]
    lease --> install["InstalledNpm.prepare<br/>npm ci and stable manifests"]
    install --> ready["Frontend[InstalledNpm]"]
    ready -->|"generate using Rust schema"| generated["Frontend[Generated]"]
    generated -->|"compare with source index"| verified["Frontend[Verified]"]
    ready -->|"admit checked-in artifacts"| existing["Frontend[ApiArtifacts]"]
    generated -->|"build"| built["Frontend[Built]<br/>dependencies and bundle retained"]
    verified -->|"build"| built
    existing -->|"build"| built
    built --> e2e["Install E2E project from its own lockfile"]
    e2e --> browser["Prepare selected browser and run Playwright"]
```

The CLI embeds the frontend bundle, while API generation consumes a Rust
exporter. Artifact-only builds admit checked-in API artifacts before building
the CLI. Standalone generation uses `BATCHALIGN_BIN`, or the headless exporter
with `binary-entry,server` and default features disabled. API drift qualification
compares the Rust schema, API types and Effect decoders. Tauri performs fresh
generation, comparison and bundling in one prepared invocation.

`BATCHALIGN_SKIP_API_SYNC=1` remains supported by the publication script through
existing-artifact admission. The ordinary bundle target also uses checked-in
artifacts, with Vitest and TypeScript checks. Schema-only checking remains
independently available.

Cold-checkout qualification reproduced the missing-generator failure and then
passed all 86 mock browser contracts through the repaired entry point. Strict
mypy rejects unleased or forged preparation, building before API admission, verification
before generation and smoke execution before building. A focused POSIX adapter contract observes kernel exclusion and descriptor-close
release. Windows lock behavior retains separate runtime qualification. The
release gate runs the same owner, whose unconditional `npm ci` establishes a fresh installation
for each project. Registry access, tool output, filesystem changes and browser
behavior retain external qualification.

## Scoped connections and observation ownership

`ConnectionPlan.prepare` admits server addresses and deduplicates their
canonical WebSocket URLs. Fleet discovery, server-list publication and socket
work share one React-owned Effect fiber. Unmount interrupts the discovery
request, connection acquisition, active reads, heartbeat or retry delay.

Effect's socket reader acquisition waits for an actual open event.
`OpenedConnection` retains that reader and its scoped writer before permitting
message handling and heartbeats. An opened connection resets the existing
one-second retry delay, and failed acquisitions back off to thirty seconds.
Effect's socket adapter applies its ten-second handshake timeout. Each attempt
finishes its scope cleanup before the next attempt starts.

```mermaid
stateDiagram-v2
    state Active {
        [*] --> Discovering
        Discovering --> Prepared: admit and deduplicate addresses
        Prepared --> Connecting: acquire scoped reader
        Connecting --> Opened: retain open reader and writer
        Connecting --> Backoff: acquisition fails
        Opened --> Backoff: close failed connection scope
        Backoff --> Connecting: retry delay completes
    }
    Active --> Closed: owner interrupts scope
    Closed --> [*]
```

A `DashboardUpdate` has a private constructor and carries both decoded payload
and producing `DashboardServer`. Reconciliation accepts that receipt, so an
arbitrary message object cannot enter through the checked consumer API. The
WebSocket envelope is declared beside the decoder, and its nested job, file
and health payloads use the generated schemas. Malformed frames are diagnosed
without publishing replacement data. Closed adapters retain Effect's inert
late-error listener to handle a browser abort event.

Progress HTTP admission checks the declared payload, requested job identity,
nonnegative completion count and unique filenames. `ObservedProgress` carries
the completed file index and classified status into the view. Presentation
reads those facts. One exhaustive classifier gives active and terminal
statuses distinct payload types, including interrupted and failed-writeback
outcomes, and is shared by production progress and polling fixtures.

Private constructors and scoped requirements constrain admission and resource
use. TypeScript permits aliases, so runtime scope finalization enforces cleanup
when callers interrupt work. React Query owns HTTP caching and request retry
policy. The connection program owns socket reconnection and heartbeat work.

## Vitest contracts and browser journeys

Run `npm test` in frontend for the retained wire, policy and scoped-effect
contracts. `@effect/vitest` supplies a fresh scope and test clock. The shared
socket fixture enters through Effect's production constructor service and
provides separate connecting, opened and observed-closed peer capabilities.
Readiness uses a queue, and lifecycle checks advance the test clock.
`tests/type-contracts.ts` is checked by TypeScript and establishes rejected
admission constructions without executing runtime tests.

Review each retained test under the typestate rubric before extending it.
Record Guards, Kind, Not a type, State and Concurrency. Runtime contracts cover
remote bytes, real effects and semantic choices remaining after type design.
The old pure Playwright polling module now runs under Vitest, and its redundant
active-state matrix was removed after sharing the typed classifier.

The shared `make batchalign-dashboard-build` target runs Vitest, static checks
and the production build for local and hosted consumers. Playwright continues
to exercise assembled browser journeys and native-adapter doubles. Packaged
Tauri qualification covers the actual native application and IPC boundary.

## Web Dashboard Development

The dashboard is a React SPA that proxies API calls to a running batchalign3
server. You need **two terminals**: one for the Rust server, one for the Vite
dev server.

### Terminal 1: Start the Rust server

```bash
# Debug build (faster compile, slower runtime: fine for dashboard dev)
cargo run -p batchalign -- serve start --foreground --port 8000
```

Or if you already have a release binary:

```bash
./target/release/batchalign3 serve start --foreground --port 8000
```

The server must be on port 8000, the Vite dev config in `frontend/vite.config.ts`
proxies `/jobs`, `/health`, and `/ws` to `localhost:8000`.

### Terminal 2: Start the Vite dev server

```bash
cd frontend
npm ci          # Install the locked development dependencies
npm run dev
```

Vite starts on `http://localhost:5173`. Open:

- **`http://localhost:5173/dashboard`**: fleet monitoring (job list, workers,
  memory, pipeline stages)
- **`http://localhost:5173/dashboard/jobs/<id>`**: job detail with file-level
  progress
- **`http://localhost:5173/dashboard/visualizations`**: algorithm trace
  visualizations (DP alignment, retokenization, FA timeline, ASR pipeline)
- **`http://localhost:5173/process`**: end-user processing flow (desktop-oriented,
  but works in browser for layout testing)

Changes to `frontend/src/` hot-reload instantly. Changes to Rust types require
rebuilding the server and regenerating OpenAPI types
(`bash scripts/generate_dashboard_api_types.sh`).

### Production mode (no dev server)

In production, the built SPA is served directly by the Rust server via
`ServeDir` with SPA fallback. After building:

```bash
scripts/build_react_dashboard.sh    # builds to ~/.batchalign3/dashboard/
```

The dashboard is then available at `http://localhost:8000/dashboard` (same
origin as the API, no proxy needed).

### Build production artifact:

```bash
scripts/build_react_dashboard.sh
```

By default, artifacts are copied to:

`~/.batchalign3/dashboard/`

Override target path:

```bash
scripts/build_react_dashboard.sh /tmp/batchalign-dashboard
```

## Desktop Dashboard (Tauri)

The native shell stays deliberately small. See the
[desktop user guide](../user-guide/desktop-app.md) for the processing flow and
`apps/dashboard-desktop/README.md` for build, qualification and coordinated
CLI/app deployment requirements.

### Runtime and ownership

`DesktopBootstrap` completes native service discovery before server consumers
mount. The shell invokes the installed CLI's `serve status --format json` and
admits only complete local observations. A stopped service may transition
through `serve start` to another observation. Missing, refused and failed
observations are visible; no default endpoint is substituted. The app owns no
service PID, has no stop command and does not stop shared work on exit.

`DesktopEndpoint` owns the admitted native origin independently of the route
URL. The bootstrap binds it before mounting queries. Redirecting `/` to
`/process`, following a navigation link or dropping query parameters cannot
substitute another origin. An unadmitted native runtime has no endpoint; there
is no guessed-port fallback. Browser query overrides remain a web-only path.

The native command inventory includes `discover_files`, `admit_job_paths`, `open_output_directory`,
`server_status`, `start_server`, `is_first_launch`, `read_config` and
`write_config`. `desktop/protocol.ts` pairs command names with their request and
response types. `DesktopContext.tsx` exposes focused capabilities; `lib/tauri.ts`
owns the dynamic native adapter. Only the dialog plugin is needed. Opening
output consumes a native-admitted existing directory.

Settings and CLI setup share `batchalign_types::asr_preferences::PreferenceEdit`.
A validated edit owns only the selected engine and an explicitly supplied Rev
key. Other configuration bytes survive an atomic publication; malformed or
unreadable configuration is refused rather than overwritten.

The frontend config capability returns admitted `PreferenceDraft`, accepts
nominal `PreferenceEdit`, and returns validated `SavedPreferences`. Raw native
DTOs stay at the adapter. A draft records retain/replace credential intent;
the old read key is reduced to availability and never replayed on Save.
Acknowledgement retires replacement intent and clears the editor. The native
writer still resolves retain against the current INI bytes at write time.

`SettingsSession` owns loading, load failure, editing and saving. A synchronous
operation claim blocks duplicate reads/writes; closing retires pending reads,
while an in-flight save keeps editing and close actions disabled. Reopening
mounts a fresh owner. Saved notices carry receipts and retire when editing.
The first-run wizard has separate engine, Rev.AI credential, saving and receipt
states; credential data cannot accompany a Whisper choice or fabricate Done.
Read failures and setup saves have visible retry paths. The first-launch check
admits an actual boolean; transport failure cannot manufacture file absence.

### Processing states

`ProcessForm` owns choosing, configuring, picking a folder, scanning inputs and
the accepted/unconfirmed result views. `JobLaunch` owns preparation and sending;
there is no parallel mutation busy flag or form admission/submission state.
A selection attempt owns
its configuration and is claimed synchronously before opening a dialog.
Only an input attempt can enter scanning. Pending selection disables the
configuration fieldset and cannot start admission. Cancel restores the earlier
configuration; input failure clears its input; Back or unmount withdraws the
owner before delayed results can settle. `FolderPicker` only presents this
form-owned state and has no independent busy state or async selection callback.

Native discovery enters `DiscoveredInput`: either a genuine empty folder or
immutable nonempty `SelectedInputs`. `PreparedJob.prepare` requires the latter,
so it has no empty-list fallback. `CommandDef` fixes each command's input kind;
one policy supplies both native discovery extensions and displayed guidance.
`InputFilePage` admits bounded pages belonging to that selected population;
rediscovery retires the previous page, and reviewing never filters submission.

`PreparedJob` captures immutable paths and validated scalar
choices before any await. Native `PathProposal::admit` is the only producer of
`AdmittedPathPlan`: it resolves directory aliases, checks source/output file
identities and admits disjoint separate roots and unique portable output names.
Only its complete receipt can create the private `JobPlan` accepted by HTTP
submission. This is read-only filesystem observation, not a file reservation.
A synchronous available-state claim prevents duplicate submissions throughout
admission and settings reads. `JobLaunchState` offers Start only when available;
checking-paths, reading-preferences, sending, finished and retired states carry
no launch action. `PreparingLaunch` is the only predecessor accepted by the
private sending edge. `LaunchRequest` distinguishes commands with explicit ASR
selection from those without it. Discovery skips nested links
and special files rather than following them beyond the selected tree.

`SubmissionOutcome` separates accepted, refused and unconfirmed HTTP results.
Only a validated `SubmittedJob` supplies the identifier used to open progress;
its receipt does not pretend to contain the full `JobInfo`. Known request and
admission refusals restore configuration, while transport loss, timeouts, server
failures, malformed JSON and unusable identifiers retain the admitted plan in
an unconfirmed state. Unexpected post-attempt failures take that same edge.
Native/preferences failures before the POST remain correctable configuration
errors. No automatic POST retry is offered; list refresh performs reads only. Recovery
offers inspection and deliberate fresh configuration, not replay of the plan.
This does not add server idempotency or persist a deduplication record across
navigation, app restarts or other clients.

`useSubmitJob` attaches the launch owner through a layout effect, so unmount
retires it at commit rather than waiting for passive cleanup. Each reply must
still own its exact phase before another native operation or the POST can begin;
a later attachment cannot revive an old snapshot. Claiming and initiating HTTP
are synchronous before notification. After sending, unmount only retires the
view: it does not cancel the server job or retry the request. Even a retired
launch refreshes job discovery after its HTTP outcome; it cannot publish into
the replacement launcher's configuration. Browser boundary tests hold native
admission and preference replies across router unmounts, and hold a real mock
HTTP reply after its job record exists. These prove transport ownership, not
packaged OS permission or model behavior.

`useJobStream` polls the authoritative job record; its historical name does not
imply an SSE completion signal. `ObservedProgress` admits the job identity,
closed status set, file identities and counts. A failed observation is distinct
from a terminal job and never triggers submission. Success requires the full
expected file population to be present and cleanly completed; diagnosed,
interrupted and writeback-failed outcomes remain visible.

`useJobCancellation` retains one `CancellationAction` per job and service origin
above the transient observation view. Available/failed states expose request or
retry; requesting and acknowledged states expose neither. Synchronous state
publication owns the double-click boundary without a separate boolean latch.
`CancellationReceipt` admits only the route's cancelled/completed/failed outcome
and its explanation. This receipt is not `ObservedProgress`; the complete job
record still owns terminal presentation. A progress outage does not reopen an
acknowledged action, and a retry retires the previous cancellation error.

### Reopened job output access

The job-page controller projects a closed loading/unavailable/ready state. Only
ready carries detail and its server/action scope. A resolved record wins over
a pending background owner lookup, so late REST/WebSocket discovery cannot
unmount that scope and implicitly retry an errored folder query. Route absence
creates no query scope; a selected job owns a keyed controller rather than
passing an empty-string job sentinel through the hooks. Controlled browser
coverage holds both initial delivery paths and the later owner lookup reply.
A failed discovery offers an explicit read retry; it cannot certify job absence
or resubmit processing work.

The loopback-only `GET /jobs/{job_id}/output-folders` projection reads the
persisted paths-mode mapping through the existing store/backend. It does not
change the database schema, reconstruct destinations from `source_dir`, or
add a second client-side job store. `LocalOutputRequest` admits peer locality
before the route can look up a job. `LocalOutputAccess` distinguishes content
mode, absent evidence, malformed records and a nonempty set of recorded folder
parents. Distinct destinations stay distinct; there is no guessed common root.
The store projects the record's identity and output evidence together as
`JobOutputFolders`; the route emits that snapshot without attaching a separately
supplied identity to its folders.

`desktopEndpointForServer` gives `LocalJobOutputRequest` the native discovery
capability only when the viewed job belongs to that exact managed origin.
`RecordedOutputFolder` requires that local request capability, rather than a
bare job-id string, to admit the response's job identity and wire states before
exposing an open action. `OutputFolderAction` owns idle/opening/failed
transitions; query and action owners are scoped to `(server, job_id)`, so stale
replies cannot settle another route. The existing native directory constructor
checks actual filesystem existence when clicked. A recorded folder is neither
a completion certificate nor a filesystem permission guarantee. Browser and
remote-job views do not borrow the local native opener.

### Verification boundaries

Run native shell tests with
`cargo test -p batchalign-dashboard-desktop --bin batchalign-dashboard-desktop`.
Shared preferences tests belong to `batchalign-types`; the CLI observation
subprocess test belongs to `cli_integration_suite`. Frontend seam checks in
`frontend/e2e/tests/mock-server.spec.mjs` fake the native transport. They cannot
certify native dialogs, filesystem permissions, model execution, signing or
installation. Qualify those with a packaged application and an identified
managed service before distribution.

## API Contract Discipline

Rust OpenAPI remains canonical. Regenerate dashboard artifacts with:

```bash
scripts/generate_dashboard_api_types.sh
```

Verify no drift (CI gate):

```bash
scripts/check_dashboard_api_drift.sh
```

### Known open work: a branded server-time type

`compareTimesNewestFirst` in `frontend/src/utils.ts` sorts jobs newest first
by comparing two server times as strings. That is correct only because both
are `MachineTime`: RFC 3339 in UTC with exactly three fractional digits, so
string order is time order. Its six callers (two in `src/state.ts`, four in
`src/components/JobList.tsx`) all pass `submitted_at`, which is a
`MachineTime`. But the signature takes `(a: string, b: string)`, so a
display-formatted time, or one with a different offset or precision, would
type-check and sort wrongly without complaint.

The generated Effect decoder currently admits this schema as a string. A future
change should put the exact UTC/precision constraint in the producing schema
and derive a branded decoder result from that constraint. The comparison API
can then require that admitted time type. The decoder generator and the static
OpenAPI generator must carry the same contract into their respective consumers.

## Dashboard Component Map

### Pages

| Route | Component | Purpose |
|-------|-----------|---------|
| `/dashboard` | `DashboardPage` | Two-column: job list + system panels |
| `/dashboard/jobs/:id` | `JobPage` → `JobDetailPageView` | Full job detail with file table |
| `/dashboard/visualizations` | `VisualizationsIndex` | Algorithm trace landing |
| `/dashboard/visualizations/:type` | `DPAlignmentPage`, etc. | Individual visualizations |
| `/process` | `ProcessPage` | Desktop processing flow |

### Job Display Components

| Component | File | Role |
|-----------|------|------|
| `JobList` | `JobList.tsx` | Renders sorted list of `JobCard`s |
| `JobCard` | `JobCard.tsx` | Single job summary: command badge, status, progress bar, metadata |
| `JobDetailPageView` | `JobDetailPageView.tsx` | Full detail: metadata grid, progress, file table, errors |
| `FileTable` | `FileTable.tsx` | Directory-grouped file rows with status, stage, progress |
| `PaginatedFileList` | `PaginatedFileList.tsx` | Wraps `FileTable` with pagination controls |
| `FilterTabs` | `FilterTabs.tsx` | Status filter tabs (All/Processing/Done/Error/Queued) + search |
| `ProgressBar` | `ProgressBar.tsx` | Animated fill bar with striped/indeterminate modes |
| `PipelineStageBar` | `PipelineStageBar.tsx` | 5-segment phase indicator (Read/Transcribe/Align/Analyze/Finalize) |
| `StatusBadge` | `StatusBadge.tsx` | Colored status pill |
| `ErrorPanel` | `ErrorPanel.tsx` | Error groups by failure category with expandable file lists; groups, labels, colours and banners come from `errorCategories.ts`, a `Record` over the generated `FailureCategory` union, so a new server category fails type-checking until it has a group |
| `ErrorCodeGroup` | `ErrorCodeGroup.tsx` | Single error code bucket |
| `StatusSummaryStrip` | `StatusSummaryStrip.tsx` | Inline done/error/active counts |
| `ActionButtons` | `ActionButtons.tsx` | Cancel/restart/delete controls |

### System Health Components (Right Column)

These panels live in the dashboard right column and consume `HealthResponse`
data from the Zustand `healthMap`:

| Component | File | Data Fields Used |
|-----------|------|-----------------|
| `WorkerProfilePanel` | `WorkerProfilePanel.tsx` | `live_worker_keys`, `live_workers` |
| `MemoryPanel` | `MemoryPanel.tsx` | `system_memory_total_mb`, `system_memory_available_mb`, `system_memory_used_mb`, `memory_gate_threshold_mb`, `memory_gate_aborts` |
| `VitalsRow` | `VitalsRow.tsx` | `worker_crashes`, `forced_terminal_errors`, `memory_gate_aborts`, `attempts_started`, `attempts_retried`, `deferred_work_units` |

The `DashboardPage` wires these by reading the first server's health from
`useStore((s) => s.healthMap)`. When a server filter is active, it uses that
server's health.

### Pipeline Phase Mapping

`PipelineStageBar` groups the 23 `FileProgressStage` enum variants into 5
visual phases. The mapping is defined in `PHASES` at the top of the component:

| Phase | Color | Stages |
|-------|-------|--------|
| Read | zinc | `reading`, `resolving_audio`, `checking_cache` |
| Transcribe | emerald | `transcribing`, `recovering_utterance_timing`, `recovering_timing_fallback` |
| Align | indigo | `aligning`, `applying_results` |
| Analyze | violet | `analyzing_morphosyntax`, `segmenting_utterances`, `translating`, `resolving_coreference`, `segmenting`, `analyzing`, `comparing`, `benchmarking` |
| Finalize | amber | `post_processing`, `building_chat`, `finalizing`, `writing` |

When adding a new `FileProgressStage` variant in Rust, add it to the
appropriate phase set in `PipelineStageBar.tsx` and to `PROGRESS_STAGE_LABELS`
in `frontend/src/utils.ts`.

### Worker Key Parsing

`WorkerProfilePanel` parses the `live_worker_keys` strings from the health
endpoint. The format is:

```text
profile:<gpu|stanza|io>:<lang>[:<engine_overrides>] (<N> total, <M> idle|shared)
```

Examples:
- `profile:gpu:eng (1 total, shared)`: one shared GPU worker for English
- `profile:stanza:eng (2 total, 1 idle)`: two Stanza workers, one currently checked out
- `profile:io:fra:{"translate":"seamless"} (1 total, 1 idle)`: IO worker with engine override

The regex parser in `parseWorkerKey()` handles these formats. If the server
changes the key format, the parser must be updated to match.

### Memory Panel Thresholds

`MemoryPanel` uses three proximity levels relative to the gate threshold:

- **safe** (green): `available > threshold × 4`
- **warning** (amber): `threshold × 2 < available ≤ threshold × 4`
- **danger** (red): `available ≤ threshold × 2`

The gate threshold marker on the gauge bar is positioned at
`(total - threshold) / total × 100%`: the point where available memory
equals the threshold.

## Dashboard State Boundary

The current dashboard intentionally splits client state by responsibility:

- Zustand owns fleet summary state such as job rows, health, connection status,
  and filter UI state.
- React Query owns REST-shaped detail payloads such as one fully resolved job
  record or trace download response.

The job detail page should surface the original submitted command options from
the REST payload. That operator-facing context is important for debugging
engine selection, incremental flags, and rerun behavior without reconstructing
the submission from logs.

Per-file progress data now also follows the same typed-boundary rule:

- `progress_stage` is the stable machine-readable stage code from Rust.
- `progress_label` is the derived operator-facing text shown in the UI.

Dashboard code should prefer `progress_stage` for client logic and treat the
label as a rendering convenience rather than a parsing target.

That split matters because the WebSocket reconciliation layer can patch the
same React Query cache entries that the detail route reads, instead of copying
one detailed job payload through a second global store slot.

## Dashboard E2E

Run dashboard Playwright smoke tests:

```bash
cd frontend
npm run e2e:install
npm run test:e2e
```

Run real-server dashboard E2E:

```bash
BATCHALIGN_REAL_SERVER_E2E=1 scripts/run_react_dashboard_smoke.sh
```

## E2E Testing Entry Points

Run the Make commands from the repository root.

| Command | Qualification |
| --- | --- |
| `make batchalign-dashboard-e2e` | Locked installation, API generation, bundle and mock browser journeys |
| `make batchalign-dashboard-e2e-real` | The same preparation and mock journeys, plus managed server journeys |
| `make batchalign-dashboard-build` | Checked-in API artifact admission, Vitest, TypeScript and bundle |
| `make batchalign-dashboard-api-check` | Fresh generation and source comparison |

Select a prepared CLI with `BATCHALIGN_BIN=/path/to/batchalign3`. Generation
otherwise builds the headless exporter. Real-server CI consumes its workflow's
CLI and wheel and supplies `BATCHALIGN_PYTHON`. The experimental desktop workflow
builds bundles through Tauri's shared frontend preparation hook.

`BATCHALIGN_PLAYWRIGHT_WITH_DEPS=1` installs Chromium's system dependencies.
`BATCHALIGN_SKIP_BROWSER_INSTALL=1` selects an already installed browser.
The publication script alone retains `BATCHALIGN_SKIP_API_SYNC=1`.

For headed iteration after preparation, use
`npm --prefix frontend/e2e run test:headed`. `frontend/E2E_ENTRY_POINTS.md`
lists the supported adapters.

## Data Flow

### Dashboard (fleet monitoring)

1. **Init**: Resolve the server URL from runtime config
2. **WebSocket**: Connect to the server and receive snapshot + real-time updates
3. **REST**: React Query fetches job lists and details
4. **State**: Zustand store tracks dashboard and connection state
5. **Updates**: WebSocket patches store + query cache in real time

Detail pages follow the same split:

- `pages/JobPage.tsx` is the route shell
- `hooks/useJobPageController.ts` owns job lookup, server resolution, and store sync
- `components/JobDetailPageView.tsx` owns detail presentation and file filters

### Process flow (desktop)

1. Discover the CLI-managed local service before mounting server consumers.
2. Detect first-run absence or show configuration read errors.
3. Select a command, input directory, explicit output policy and parameters.
4. Claim a `PreparedJob`, await native path admission, then construct `JobPlan`.
5. Submit with `paths_mode: true` and retain the returned job identity.
6. Poll that job's record; reconnect without resubmitting after an outage.
7. Present its actual terminal state, diagnostics and output directory.
