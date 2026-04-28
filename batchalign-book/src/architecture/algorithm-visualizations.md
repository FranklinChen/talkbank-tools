# Algorithm Visualizations

**Status:** Phase 2 complete (DP Alignment live mode + structured result refactor);
Phases 3–4 in progress

The batchalign3 dashboard includes interactive algorithm visualizations that
show the internal workings of key algorithms — DP alignment, ASR
post-processing, forced alignment timing, and retokenization mapping.  Each
visualization supports two modes:

- **Static mode** — educational, with editable sample data and no server
  required.  TypeScript ports of the Rust algorithms run locally in the browser.
- **Live mode** — shows actual intermediate states from a completed job,
  fetched via the `GET /jobs/{id}/traces` REST endpoint.

## Architecture

```
┌─────────────────────────────────────┐
│ React Dashboard (frontend/)         │
│                                     │
│  /dashboard/visualizations/         │
│    ├── dp-alignment   ──┐           │
│    ├── asr-pipeline   ──┤ Static    │
│    ├── fa-timeline    ──┤ sample    │
│    └── retokenize     ──┘ mode      │
│                                     │
│  /dashboard/jobs/:id/traces/        │
│    ├── dp-alignment   ──┐           │
│    ├── asr-pipeline   ──┤ Live      │
│    ├── fa-timeline    ──┤ job       │
│    └── retokenize     ──┘ mode      │
│                                     │
│  engines/  ← TS ports for static    │
│  mode + rendering logic             │
└──────────────┬──────────────────────┘
               │ REST (live mode)
               ▼
┌─────────────────────────────────────┐
│ Rust Server                         │
│                                     │
│  GET /jobs/{id}/traces              │
│    → JobTraces per file             │
│                                     │
│  Structured results:                │
│    FaResult, MorphosyntaxResult     │
│    always carry intermediate data   │
│                                     │
│  Storage: ephemeral in-memory       │
│    (moka LRU, 50 jobs, 1hr TTL)    │
└─────────────────────────────────────┘
```

## Structured Result Types

Orchestrators return rich result types that always carry intermediate data,
regardless of whether traces are stored.  The dispatch layer decides what to
persist based on the job's `debug_traces` flag.

### FaResult

Returned by `process_fa()` in `crates/batchalign-app/src/fa/`:

```rust
pub struct FaResult {
    pub chat_text: String,
    pub groups: Vec<FaGroupTrace>,
    pub pre_injection_timings: Vec<Vec<Option<TimingTrace>>>,
    pub timing_mode: FaTimingMode,
    pub violations: Vec<ViolationTrace>,
}
```

The dispatch layer extracts `chat_text` for file output.  When `debug_traces`
is enabled, it calls `into_timeline_trace()` to build a `FaTimelineTrace` and
stores it via `TraceStore::upsert_file()`.

### MorphosyntaxResult

Returned by `process_morphosyntax()` (single-file path):

```rust
pub struct MorphosyntaxResult {
    pub chat_text: String,
    pub retokenizations: Vec<RetokenizationInfo>,
}
```

`RetokenizationInfo` is emitted by `inject_results()` in
`batchalign-chat-ops` whenever Stanza retokenization occurs — it captures the
original words, Stanza tokens, and the word-to-token mapping for each affected
utterance.

### Design principle

Previous iterations passed a `debug_traces: bool` parameter through the
orchestrator call chain and conditionally collected trace data alongside the
main output.  This added complexity without benefit — the intermediate data
(groups, timings, retokenization mappings) was already computed as part of
normal processing.

The current design makes the orchestrator API surface richer by default:
structured results always carry the intermediate state.  The `debug_traces`
flag only controls whether the dispatch layer *stores* that data in the
ephemeral trace cache.  This is simpler, avoids parameter threading, and opens
the door to other consumers of the structured data (e.g. detailed error
reports, regression analysis).

## Trace Storage

`TraceStore` wraps a `moka::future::Cache<String, Arc<JobTraces>>` with:

- **Capacity:** 50 jobs (LRU eviction)
- **TTL:** 1 hour per entry
- **Location:** field on `JobStore` (accessible everywhere the store is)
- **Concurrency:** uses moka's `and_upsert_with` for per-key atomic
  read-modify-write — concurrent FA file completions for the same job are
  serialized without blocking unrelated jobs

Traces are diagnostic-only and not persisted to SQLite.

The primary write API is `upsert_file(job_id, file_index, file_traces)` which
atomically gets-or-creates the `JobTraces` entry, inserts the file, and puts
it back.  This is safe to call from multiple concurrent `process_one_fa_file`
tasks within the same job.

### Activation

Per-job: set `"debug_traces": true` in the job submission JSON.

```json
POST /jobs  { "command": "align", "debug_traces": true, ... }
```

### REST Endpoint

```
GET /jobs/{job_id}/traces
  → 200: JobTraces JSON
  → 404: job not found
  → 204: job exists but no traces collected

GET /jobs/{job_id}/traces/{file_index}
  → 200: FileTraces JSON (single file)
  → 404: file index not found
```

## Trace Data Model

All trace types live in `crates/batchalign-app/src/types/traces.rs`.

```
JobTraces
  └── files: BTreeMap<usize, FileTraces>
        ├── filename: String
        ├── dp_alignments: Vec<DpAlignmentTrace>
        ├── asr_pipeline: Option<AsrPipelineTrace>
        ├── fa_timeline: Option<FaTimelineTrace>
        └── retokenizations: Vec<RetokenizationTrace>
```

| Trace type | Source orchestrator | What it captures |
|-----------|-------------------|-----------------|
| `DpAlignmentTrace` | `dp_align.rs` | Full cost matrix, traceback path, alignment result |
| `AsrPipelineTrace` | `transcribe.rs` | 7-stage ASR post-processing intermediates |
| `FaTimelineTrace` | `fa.rs` | Group boundaries, pre/post timings, violations |
| `RetokenizationTrace` | `morphosyntax.rs` | Word↔token mapping per utterance |

## Frontend

### Visualizations

| Visualization | Route (static) | Route (live) | Status |
|--------------|----------------|-------------|--------|
| DP Alignment Explorer | `/dashboard/visualizations/dp-alignment` | `/dashboard/jobs/:id/traces/dp-alignment` | Complete |
| Retokenization Mapper | `/dashboard/visualizations/retokenize` | `/dashboard/jobs/:id/traces/retokenize` | Static complete |
| ASR Pipeline Waterfall | `/dashboard/visualizations/asr-pipeline` | — | Planned |
| FA Timeline | `/dashboard/visualizations/fa-timeline` | — | Planned |

### TypeScript Engine Ports

Static mode uses TypeScript ports of the Rust algorithms located in
`frontend/src/engines/`:

| Engine file | Rust source | What it ports |
|-------------|------------|--------------|
| `dpAlignment.ts` | `batchalign-chat-ops/src/dp_align.rs` | `align_small` with step-by-step emission |
| `retokenize.ts` | `batchalign-chat-ops/src/retokenize/mapping.rs` | Word↔token mapping |

These are faithful ports — same algorithm, same cost model, same edge cases —
not approximations.

### Dual-Mode Pattern

Each visualization page accepts a route parameter `/:id` for live mode.  When
present, it fetches traces from the server via `useTraceQuery(id)`.  When
absent, it uses local state and the TS engine for static mode.

```tsx
function DPAlignmentPage() {
  const { id } = useParams();
  const { data: traces } = useTraceQuery(id);  // live mode

  const dpResult = useMemo(() => {
    if (id) {
      // Convert server trace to visualization format
      return traceToResult(traces.dp_alignments[selectedIdx]);
    }
    // Static mode: run TS engine locally
    return alignWithSteps(payload, reference, matchMode);
  }, [id, traces, payload, reference, matchMode]);

  // Same visualization components for both modes
  return <CostGrid ... />;
}
```

### Shared Components

Reusable visualization components in `frontend/src/components/visualizations/`:

| Component | Purpose |
|-----------|---------|
| `CostGrid` | SVG grid for DP cost matrix with fill/traceback animation |
| `StepControls` | Play/pause/step/skip controls for stepping through algorithm |
| `ModeToggle` | Static ↔ Live mode indicator |
| `SpanRuler` | Horizontal span bar for retokenization mapping |

## Trace Collection Points

| Orchestrator | What to capture | Where in code |
|-------------|-----------------|---------------|
| `fa.rs` | Group boundaries, pre/post timings, violations | After `parse_fa_response()` and `apply_fa_results()` |
| `morphosyntax.rs` | Retokenization mappings per utterance | Return value of `inject_results()` |
| `transcribe.rs` | ASR pipeline intermediate states | Wrap `process_raw_asr()` stages (Phase 3) |
| `dp_align.rs` | Cost matrix + traceback | Optional trace output parameter (Phase 2) |

## Implementation Phases

### Phase 1: Infrastructure + Retokenization (complete)

- Visualization routes, landing page, shared components
- Retokenization engine port and static-mode page
- `TraceStore`, trace type definitions, REST endpoint stub

### Phase 2: DP Alignment + Trace Collection (complete)

- DP alignment engine port with step emission
- `CostGrid` visualization with fill/traceback animation
- Live mode: `useTraceQuery` hook, server trace conversion
- Structured result refactor: `FaResult`, `MorphosyntaxResult`
- FA trace collection and storage in dispatch layer

### Phase 3: ASR Pipeline Waterfall (planned)

- Port ASR post-processing 7 stages to TypeScript
- `DiffView` component for stage-by-stage transforms
- Trace collection in `transcribe.rs`

### Phase 4: FA Timeline (planned)

- DAW-style SVG timeline with pan/zoom
- FA grouping and timing injection animation
- Post-processing before/after comparison
