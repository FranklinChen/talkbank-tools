# HTTP Request Body Limits

**Status:** Current
**Last updated:** 2026-03-27 22:02 EDT

## The Problem

The batchalign3 server has **two independent body-size limits** that gate
incoming HTTP requests.  Before this was understood and fixed, large batch
submissions (e.g. 50+ CHAT files in a single `POST /jobs`) silently hit the
inner limit and returned `413 Payload Too Large` even though the configurable
outer limit was generous.

## Two Layers of Limits

### Layer 1: `RequestBodyLimitLayer` (outer, configurable)

Defined in `crates/batchalign-app/src/routes/mod.rs` as the outermost
body-aware middleware:

```rust
let max_body_bytes = state.environment.config.max_body_bytes_mb.0 as usize * 1024 * 1024;
// ...
.layer(RequestBodyLimitLayer::new(max_body_bytes))
```

This is the **intended** body-size guard.  It is configured via
`max_body_bytes_mb` in `server.yaml` and defaults to **100 MB**
(`default_max_body_bytes_mb()` in `config.rs`).

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

The job router in `crates/batchalign-app/src/routes/jobs/mod.rs` applies
`DefaultBodyLimit::disable()` to all job routes:

```rust
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

The default 100 MB limit comfortably handles batches of up to ~800 files.
Operators who need larger batches can raise `max_body_bytes_mb` in
`server.yaml`.

## Future Work

- **TODO:** The 100 MB default was chosen ad hoc.  Revisit whether this should
  be higher by default, or whether the CLI should automatically chunk large
  submissions into multiple jobs rather than relying on the server to accept
  arbitrarily large payloads.
- **TODO:** Consider logging which limit was hit (inner vs outer) to make
  debugging easier if someone re-introduces an inner limit.
- **TODO:** The `max_body_bytes_mb` config key applies globally to all routes.
  If other routes (e.g. bug report submission) need tighter limits, we may
  want per-route configuration.

## Related Files

| File | Role |
|------|------|
| `crates/batchalign-app/src/routes/mod.rs` | Outer `RequestBodyLimitLayer` |
| `crates/batchalign-app/src/routes/jobs/mod.rs` | Inner limit disabled via `DefaultBodyLimit::disable()` |
| `crates/batchalign-app/src/types/config.rs` | `max_body_bytes_mb` field and default |
| `crates/batchalign-app/src/types/request.rs` | `JobSubmission` and `FilePayload` structs |
