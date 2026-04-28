# Parameter Design

**Status:** Current
**Last verified:** 2026-03-11

The server-side orchestrators (`morphosyntax.rs`, `fa.rs`, `transcribe.rs`,
`compare.rs`) process CHAT files through multi-step pipelines that require many
configuration values. Without deliberate design, function signatures accumulate
10–16 parameters, creating boolean blindness and making call sites unreadable.

This document describes the parameter type system in `batchalign-app` and the
principles behind it.

The companion page [Wide Struct Audit](wide-structs.md) covers the same design
pressure from the data-shape side: large field bags, boolean-heavy structs, and
where wide boundary types are acceptable versus where they are not.

## The Problem

A pre-refactoring FA function signature looked like this:

```rust
pub async fn process_fa(
    chat_text: &str,
    audio_path: &str,
    audio_identity: &AudioIdentity,
    total_audio_ms: Option<u64>,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &str,
    lang: &str,
    timing_mode: FaTimingMode,
    max_group_ms: u64,
    engine: FaEngineType,
    override_media_cache: bool,    // ← what does "true" mean?
    write_wor: bool,         // ← what does "false" mean?
    progress: Option<&ProgressSender>,
) -> Result<FaResult, ServerError>
```

14 parameters. Two booleans whose meaning requires reading the implementation.
Three audio-related values always passed together. Three infrastructure values
(`pool`, `cache`, `engine_version`) repeated across every orchestrator.

## Boolean Blindness Elimination

### `CachePolicy`

Every NLP orchestrator accepts a cache policy. The old `override_media_cache: bool`
parameter inverted the natural reading — `true` meant *skip* the cache, not
*use* it.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Use the cache normally (check for hits, store new results).
    UseCache,
    /// Skip cache lookups (always recompute; still stores results).
    SkipCache,
}

impl CachePolicy {
    pub fn should_skip(&self) -> bool {
        matches!(self, Self::SkipCache)
    }
}
```

Call sites read naturally:

```rust
// Before
if override_media_cache { /* skip */ } else { /* use */ }

// After
if cache_policy.should_skip() { /* skip */ } else { /* use */ }
```

### `WorTierPolicy`

FA processing optionally generates `%wor` tiers (word-level timing). The old
`write_wor: bool` is replaced by:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorTierPolicy {
    Include,
    Omit,
}

impl WorTierPolicy {
    pub fn should_write(&self) -> bool {
        matches!(self, Self::Include)
    }
}
```

### `From<bool>` Bridge

Both enums implement `From<bool>` for the boundary where options are
deserialized from CLI flags or JSON:

```rust
impl From<bool> for CachePolicy {
    fn from(override_media_cache: bool) -> Self {
        if override_media_cache { Self::SkipCache } else { Self::UseCache }
    }
}
```

The conversion happens once at the dispatch layer. Interior code never sees
bare booleans.

## Parameter Structs

### `PipelineServices`

Infrastructure references needed by every orchestrator.

```rust
#[derive(Clone, Copy)]
pub(crate) struct PipelineServices<'a> {
    pub pool: &'a WorkerPool,
    pub cache: &'a UtteranceCache,
    pub engine_version: &'a EngineVersion,
}
```

`Clone + Copy` because all fields are references. Constructed once per dispatch
and threaded through the orchestrator chain. Note that `engine_version` is
`&EngineVersion` (not `&str`) — the newtype propagates through the cache layer.

### `MorphosyntaxParams`

Groups the five parameters specific to morphosyntax processing:

```rust
pub struct MorphosyntaxParams<'a> {
    pub lang: &'a LanguageCode3,
    pub tokenization_mode: TokenizationMode,
    pub cache_policy: CachePolicy,
    pub multilingual_policy: MultilingualPolicy,
    pub mwt: &'a MwtDict,
}
```

Note `lang` is `&LanguageCode3` (not `&str`), preventing confusion with other
string parameters. The `LanguageCode3` deref-coerces to `&str` at the few
points where a raw string is needed (e.g., passing to `LanguageCode::new()`).

### `FaParams`

Groups the five FA-specific processing parameters:

```rust
#[derive(Debug, Clone, Copy)]
pub struct FaParams {
    pub timing_mode: FaTimingMode,
    pub max_group_ms: u64,
    pub engine: FaEngineType,
    pub cache_policy: CachePolicy,
    pub wor_tier: WorTierPolicy,
}
```

`Clone + Copy` because all fields are small values. Constructed in the dispatch
layer from `CommandOptions::Align`.

### `AudioContext`

Groups the three audio-related values always passed together:

```rust
pub struct AudioContext<'a> {
    pub audio_path: &'a Path,
    pub audio_identity: &'a AudioIdentity,
    pub total_audio_ms: Option<u64>,
}
```

Note `audio_path` is `&Path` (not `&str`) — file paths use `std::path` types
throughout the Rust domain code. Conversion to `String` happens only at the
IPC boundary when serializing JSON for Python workers (`path.to_string_lossy()`).

## Result: Reduced Signatures

| Function | Before | After |
|----------|--------|-------|
| `process_morphosyntax` | 9 params | 3 (`chat_text`, `services`, `params`) |
| `process_morphosyntax_incremental` | 10 params | 4 (`before`, `after`, `services`, `params`) |
| `process_fa` | 14 params | 5 (`chat_text`, `audio`, `services`, `fa_params`, `progress`) |
| `process_fa_incremental` | 15 params | 6 |
| `process_compare` | 8 params | 6 |
| `process_transcribe` | 6 params | 4 (`audio_path`, `services`, `opts`, `progress`) |
| `process_one_transcribe_file` | 16 params | 8 |

## Where Grouping Doesn't Help

Dispatch-level functions (`dispatch_fa_infer`, `dispatch_transcribe_infer`)
still carry 8–11 parameters and retain `#[allow(clippy::too_many_arguments)]`.
These are multi-concern routers that:

1. Take `Arc<WorkerPool>` and `Arc<UtteranceCache>` (not references) because
   they clone into spawned `JoinSet` tasks.
2. Carry job identity (`job_id`, `correlation_id`, `store`) alongside
   processing parameters.
3. Read command-specific options from the job store, then construct the
   parameter structs described above for the inner orchestrators.

The parameter structs live at the orchestrator level. The dispatch level is
intentionally a thin translation layer between job-store state and typed
orchestrator parameters.

The same rule now applies inside the server store layer. SQLite write-through
helpers use named records such as `PersistedJobUpdate`, `PersistedFileUpdate`,
`AttemptStartRecord`, and `AttemptFinishRecord` instead of long ordered
argument lists, so persistence boundaries stay explicit even inside internal
control-plane code.

## Layered Conversion

```
CLI flags / JSON body           ← bool, String, numbers
    │
    ▼
CommandOptions (options.rs)     ← deserialized, still bool for override_media_cache
    │
    ▼
Dispatch layer (infer.rs)       ← CachePolicy::from(opts.override_media_cache)
    │                              FaParams { ... }
    │                              MorphosyntaxParams { lang: &lang, ... }
    │                              PipelineServices { engine_version: &ev, ... }
    ▼
Orchestrator (fa.rs, etc.)      ← typed params: &LanguageCode3, &EngineVersion,
    │                              &Path, CachePolicy — no bare strings or bools
    ▼
Pipeline internals              ← params.cache_policy.should_skip()
    │                              params.lang.as_ref() (where &str needed)
    ▼
IPC boundary (worker JSON)      ← path.to_string_lossy(), &*lang (Deref)
```

Raw primitives enter at the CLI/JSON boundary. The dispatch layer converts
booleans to enums, strings to newtypes, and string paths to `PathBuf`. From
the dispatch layer inward, all code uses typed parameters. At the IPC boundary
(JSON for Python workers), newtypes deref-coerce to `&str` and paths convert
via `to_string_lossy()`.

## Files

| File | Contents |
|------|----------|
| `types/params.rs` | `CachePolicy`, `WorTierPolicy`, `MorphosyntaxParams`, `FaParams`, `AudioContext` |
| `pipeline/mod.rs` | `PipelineServices` |
| `transcribe.rs` | `TranscribeOptions` |

## Guidelines

1. **No bare booleans in orchestrator signatures.** If a boolean controls
   behavior, wrap it in an enum.
2. **Group parameters that are always passed together.** If three values
   always appear as consecutive function arguments, they belong in a struct.
3. **Convert at the boundary.** `From<bool>` impls bridge CLI/JSON booleans
   to domain enums. The conversion happens once, in the dispatch layer.
4. **Don't force grouping on dispatch functions.** Dispatch routers read
   heterogeneous state from the job store and construct typed parameters for
   orchestrators. Their parameter lists are inherently wider.
5. **Prefer references in parameter structs.** `PipelineServices<'a>` and
   `MorphosyntaxParams<'a>` borrow rather than own, avoiding unnecessary
   clones. Use `Clone + Copy` derives when all fields are references or
   small values.
