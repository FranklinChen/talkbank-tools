# HTTP Request Body Limits

**Status:** Current
**Last updated:** 2026-04-29 17:05 EDT

## The Problem

The batchalign3 server has **two independent body-size limits** that gate
incoming HTTP requests.  Before this was understood and fixed, large batch
submissions (e.g. 50+ CHAT files in a single `POST /jobs`) silently hit the
inner limit and returned `413 Payload Too Large` even though the configurable
outer limit was generous.

## Two Layers of Limits

### Layer 1: `RequestBodyLimitLayer` (outer, configurable)

Defined in `crates/batchalign/src/routes/mod.rs` as the outermost
body-aware middleware:

```rust,ignore
let max_body_bytes = state.environment.config.max_body_bytes_mb.0 as usize * 1024 * 1024;
// ...
.layer(RequestBodyLimitLayer::new(max_body_bytes))
```

This is the **intended** body-size guard.  It is configured via
`max_body_bytes_mb` in `server.yaml` and defaults to **512 MB**
(`default_max_body_bytes_mb()` in `types/config/server.rs`).

### Layer 2: axum `Json` extractor (inner, was 2 MB)

Axum's `Json<T>` extractor enforces its own body limit **independently** of any
`RequestBodyLimitLayer`.  The default is **2 MB** — a safe-out-of-the-box
value for generic web applications, but far too low for batchalign's use case.

The `POST /jobs` handler uses `Json<JobSubmission>` to deserialize the request.
A `JobSubmission` contains the full text content of every submitted CHAT file
(as `Vec<FilePayload>`, where each `FilePayload.content` is the raw CHAT
string).  Even a modest batch of 20 CHAT files can exceed 2 MB.

This inner limit fires **before** the outer `RequestBodyLimitLayer` gets a
chance to evaluate the request, producing an identical `413` status code.  The
error message (`"Failed to buffer the request body: length limit exceeded"`)
gives no indication which limit was hit.

### The Fix

The job router in `crates/batchalign/src/routes/jobs/mod.rs` applies
`DefaultBodyLimit::disable()` to all job routes:

```rust,ignore
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/jobs", post(submit_job))
        // ... other routes ...
        .layer(axum::extract::DefaultBodyLimit::disable())
}
```

This removes the 2 MB `Json` extractor limit entirely.  The outer
`RequestBodyLimitLayer` remains as the sole body-size guard, governed by the
`max_body_bytes_mb` config value.

## Practical Sizing

CHAT files average ~120 KB.  JSON serialization adds minimal overhead (CHAT
text is mostly ASCII, so JSON string escaping is negligible).  Rough payload
sizes for batch submissions:

| Files | Approximate payload |
|------:|--------------------:|
|    10 |              ~1 MB  |
|    50 |              ~6 MB  |
|   200 |             ~25 MB  |
|   500 |             ~62 MB  |
| 1,000 |            ~120 MB  |
| 4,000 |            ~480 MB  |

The default 512 MB limit comfortably handles the largest batches the CLI
ships today (CHILDES-eng-uk, CHILDES-other), where 500-file chunks plus
headroom were the empirical ceiling that motivated the raise from the
historical 100 MB. Operators who need larger batches can raise
`max_body_bytes_mb` in `server.yaml`.

## Operational Knobs

### Global default

`server.yaml` `max_body_bytes_mb` overrides the compile-time 512 MB default.
Setting it to a smaller value tightens the global cap for every route;
setting it larger raises the ceiling for unusually large submissions.

### Per-route limits

If a future route needs a tighter limit than the global cap (for example,
a small-payload endpoint where 512 MB is wasteful and a tight cap would
detect abuse early), the route's `Router` can wrap a per-route
`RequestBodyLimitLayer` *inside* the global one:

```rust,ignore
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/some-tight-endpoint", post(handler))
        .layer(RequestBodyLimitLayer::new(5 * 1024 * 1024)) // 5 MB
}
```

Tower-http's `RequestBodyLimitLayer` is composable: the innermost
non-zero limit on the request path wins, so per-route limits are
strictly narrower than the global one. We have not wired this for any
specific route today because every body-accepting endpoint
(`POST /jobs`, the cancel/restart variants) wants the global cap.
When a new route lands that needs a different limit, the right move
is a per-route layer, not a new top-level config knob.

### Inner-vs-outer rejection diagnosis

When a body limit fires, axum's default 413 message
(`"Failed to buffer the request body: length limit exceeded"`) does not
indicate which layer rejected the request. This was the original
debugging headache that motivated the fix above. A typed
`PayloadTooLarge { limit_layer: Inner | Outer, configured_bytes: u64 }`
error shape is planned as part of the PyO3 typed-error contract work
(see `crates/batchalign/src/error.rs`); when that lands, the same
error type will surface inner-vs-outer at the rejection site through
`tracing::warn!`. Today, the outer `RequestBodyLimitLayer` is the only
configured body limit on `/jobs`, so any 413 from that route is the
outer layer firing.

## Related Files

| File | Role |
|------|------|
| `crates/batchalign/src/routes/mod.rs` | Outer `RequestBodyLimitLayer` |
| `crates/batchalign/src/routes/jobs/mod.rs` | Inner limit disabled via `DefaultBodyLimit::disable()` |
| `crates/batchalign/src/types/config/server.rs` | `max_body_bytes_mb` field and 512 MB default |
| `crates/batchalign/src/types/request.rs` | `JobSubmission` and `FilePayload` structs |
