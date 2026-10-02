# Changelog

Fixes and behaviour changes, newest first. The book describes the current
design; this file records how it changed.

## Unreleased

### Server construction

- The prepared-worker app constructors take one named `AppStorageOverrides`
  value instead of three positional directory overrides. Path resolution,
  storage admission, prepared-worker ownership and injected-clock behavior are
  unchanged; Rust callers name the overrides at the boundary.
- Blocking registry, file-lock and handshake operations use the shared
  tracing-preserving blocking helper, retaining the caller's job/file context.

### Dashboard API

- The never-written per-file fields `error_codes` and `error_line` are gone
  from `FileStatusEntry` (and from the TUI and the dashboard, which grouped
  failures by code and showed a line nothing supplied). Every producer wrote
  `None`; the dashboard groups failures by category, labelled by the first
  file's error.
- The `/bug-reports` routes are removed. They listed a directory nothing
  writes, while the book said validation failures write reports there and
  purge the cache entries behind them; neither happens. A validation failure
  fails the file with the validation errors as its error, and the cache is
  left as it is. The dashboard's validation banner no longer says a report
  was filed. The PyO3 `CHATValidationException.bug_report_id`, which
  nothing set, is gone, and Python's `classify_error` drops the `validation`
  category it could only reach through it.

### Job store

- A job's status row is written whole from the job itself
  (`Job::status_columns`, `JobDB::write_job_status`), replacing
  `update_job_status`, which kept `error`, `completed_at` and `num_workers`
  through `COALESCE` and wrote whatever status the caller passed. A job
  requeued by startup recovery kept the error of its earlier run, in
  memory and in the row; it is now cleared. A cancel that arrived after
  the job had already finished overwrote its row with `cancelled` and a new
  completion time; the row now keeps the status the job has. The image
  (`JobStatusColumns`) is built only by the job, by the stop transition and
  by the database read of a stored row (`read_stored`); a new job's row
  takes its status as a `JobStatus` (`NewJobRecord.status` was a string).
- Stopping a job is one transition, `JobStatusColumns::stopped(Stop)`:
  shutdown and cancellation apply it to a live job (`Job::stop`, which
  adopts the whole image) and startup recovery to each queued or running
  row. Recovery wrote `status` and `completed_at` by hand and kept a
  pending retry time (`next_eligible_at`) the shutdown path clears, so one
  transition left two different rows.
- A file's `file_statuses` row is written from its whole phase on every
  write, NULLs included (`FilePhase::columns`): insertion as
  `FilePhase::Queued`, every transition, startup interruption and the
  recovery requeue. The writer used `COALESCE` on `error`,
  `error_category`, `started_at` and `finished_at`, so a file that failed
  and then succeeded was stored `done` with its old error, and every
  startup logged and dropped it; interruption was one bulk `UPDATE` and the
  requeue a hand-written reset. Startup interruption now writes each job
  and the files the interruption moves in one transaction; a failed file
  write used to leave the job `interrupted` with its file still in flight.
  `FilePhase::from_row` reads the `FilePhaseColumns` row image.
- A file failure has a private shape and a failed phase always has one.
  This build makes only `FileFailure::recorded(message, category)`; a
  stored row is read at the database boundary as a message alone, a
  category alone, or, for an `error` or retry row that named neither, an
  unrecorded failure (`of_failed_row`), so `FilePhase::Error` and
  `RetryPending` carry a `FileFailure`, not an `Option`. An interrupted
  row's columns that held neither are no failure (`from_columns`); an
  interrupted file with an empty failure used to write no column and read
  back as having none.
- Loading a row reports every column its phase does not own, by one rule for
  every phase. A `processing` row with a finish time, an `error` row with a
  retry deadline, and an `interrupted` row with either were dropped without a
  report.
- A job or file row that startup recovery had to reinterpret for this
  build's own rules is rewritten once from what was loaded: a file column
  its phase does not own, or a submitter name without an address
  (`Submitter::from_columns` returns `StoredSubmitter`, whose
  `NameWithoutAddress` is recorded as no submitter and logged; it was
  dropped silently). Each was reported at every startup and never
  repaired. A value this build cannot read (a file status or category, a
  job status, a language or a command, as a newer build writes before a
  rollback) is read with a fallback, reported, and never rewritten
  (`RecoveredFilePhase::Foreign`), so the other build's row survives;
  rewriting it as `error` was destructive. A repair write that fails is
  logged and startup continues; it used to abort startup.
- `bug_report_id` is gone: a migration drops the `file_statuses` column, and
  the field is removed from `FileStatus`, the `FileStatusEntry` API (OpenAPI
  schema and dashboard types regenerated), the CLI failure summary, the
  debug-artifact summary and `jobs` inspection. Nothing ever wrote it, so
  every reader saw `None`.

### Worker protocol

- A batch-inference item (`InferResponse`) carries its outcome as a union,
  `{"outcome": {"kind": "produced", "result": ...} | {"kind": "failed",
  "error": "..."}, "elapsed_s": ...}`, in Rust (`ItemOutcome`) and Python
  (`ItemProduced` / `ItemFailed`). It held `result` and `error` as two
  optional fields, so a response could hold both (the PyO3 normalizer kept
  the error and dropped the result) or neither (a runtime failure invented
  for the reader). "Neither" is refused too: a produced item's `result` is
  any JSON value but `null` (`ItemPayload`), and the outcome and the
  response refuse fields they do not name (`deny_unknown_fields`; Python
  `extra="forbid"` and a `null` check). The wire shape changes; the Rust
  server and the Python worker ship together.
- Batch inference items report each item's own time or `null`. `elapsed_s`
  is `null` for an item no work ran for (invalid payload, provider not
  loaded, nothing to analyze, language group failed first); Rust reads it
  as `ItemElapsed::NotExecuted`, and the key stays required. Python's
  `InferResponse.elapsed_s` lost its `0.0` default; items are built with
  `InferResponse.timed(work)` or `InferResponse.unexecuted`. The Cantonese
  ASR and FA providers, translate and coref no longer stamp the batch total
  on the first item and `0.0` on the rest: each item now reports its own
  call's time and the batch total goes to the log.
- Python batch handlers share one loop (`batchalign.worker._batch.answer_batch`):
  item validation, per-item timing, unexecuted answers (`Unexecuted`) and
  the batch-total log, for the HK ASR providers (through
  `answer_asr_batch`), coreference and Cantonese FA. A not-loaded HK ASR
  provider answers every item as a failure (`ProviderNotLoaded`); Aliyun
  raised a `RuntimeError` for the whole batch. Utseg builds its items with
  the shared `ItemProduced` / `ItemFailed` and `InferResponse.timed`; the
  `FiniteNonNegativeFloat` alias is defined once (in `_types.py`); the
  test-echo delay has one helper; test doubles no longer construct items
  with an invented `elapsed_s=0.0`. Item timing reads a module clock
  (`_types._item_clock`), so tests no longer replace `time.monotonic` for
  the whole process, and their clocks move only during the fake work.
- A utseg item (`UtsegItemResultV2`) is a union of its real states,
  `{"kind": "boundary_model", "assignments", "boundary_model_evidence"}`,
  `{"kind": "unattributed", "assignments"}`, `{"kind": "constituency",
  "trees"}` or `{"kind": "failed", "error"}`, in Rust and Python. It held
  four optional fields, and the server checked eight combinations of them at
  run time. The worker bridge parses each item through the union like the
  other text tasks, so a malformed item is that item's failure
  (`invalid utseg host item`), not the whole batch's. In the constituency
  fallback, a parse that raises, a request no Stanza pipeline could be built
  for, a parse with no tree and a tree that does not read are each the
  item's failure (`ConstituencyParse::read`; the item type requires at
  least one tree); all of them used to segment the item into one utterance
  (an empty tree list, or an unreadable tree skipped with a warning). The
  unused `compute_assignments_single` is removed. The wire shape changes;
  the Rust server and the Python worker ship together.
- An ASR element's timing is admitted once, where the provider's numbers
  are read, and travels as one value: `AsrElementV2.timing` is
  `{"kind": "timed", "start_ms", "end_ms"}` (an `AdmittedInterval`) or
  `{"kind": "untimed", "cause"}` (it was `start_s` and `end_s`, two
  optional positions the transcript side admitted again). One policy for a
  pair that is not an interval (inverted, or past the admissible range, as
  an epoch timestamp is, which an order check alone let through), on every
  route: it refuses the response, naming the element. Rev.AI transcripts
  now do too (`AdmittedRevTranscript`); their inverted or epoch-valued pairs
  reached the transcript and were dropped untimed there. UTR still discards
  such a token and counts it, now as `inadmissible` for the range; its
  millisecond conversion, and the Rev.AI timed-word projection's, saturated
  a position past `u64` instead of refusing it
  (`AudioPositionSeconds::nearest_millis`, removed). The interval bound's
  range is in the schema and the Python bound has the same maximum
  (`AdmittedMillisV2`), so an out-of-range speaker segment is refused in
  Python as it is in Rust. `UntimedCause` moves to
  `batchalign_types::interval`.
- An `{"op":"error"}` line's `kind` is required on every reader and in
  Python; a line without it is a protocol violation, not a retryable runtime
  failure. A third kind, `invalid_request`, reports a request the worker
  refused as sent (unparseable line, missing op or `request` mapping, a
  payload that failed the request model) and is a terminal
  `WorkerError::RequestRefused`. The Rust-owned request dispatcher emitted
  those refusals without `kind`, which the readers turned into a retryable
  `WorkerResponse`, so a deterministic refusal was retried. Every envelope
  is built by one PyO3 function (`error_payload`, reached from Python as
  `batchalign_core.error_envelope`); the kinds have one Python spelling,
  the extension stub's `WorkerErrorKind`, checked against Rust's list
  (`batchalign_core.WORKER_ERROR_KINDS`). The sequential TCP handle uses
  the stdio handle's request and response envelopes instead of its own
  copies.
- Shared GPU dispatch fails with the same `WorkerError` as sequential
  dispatch when a worker's reply cannot answer a request: a reply the
  protocol refuses is `WorkerError::Protocol` (not retried), and an
  `op=error` line maps by its required `kind` (`runtime` retried,
  `bootstrap` and `invalid_request` terminal). The GPU reader answered all
  of these with a failed `ExecuteResponseV2` it composed itself (code
  `invalid_payload`), which the retry loop never saw, and it ignored
  `kind`. Worker restart differs on purpose: a sequential handle is
  discarded after an undecodable reply, a shared GPU worker is kept.
- The stdio and TCP shared GPU workers share one transport,
  `SharedGpuChannel`. Changes in behaviour:
  - A failure line reaches only its owner. A refused `execute_v2` line or an
    untagged `op=error` that names no request used to fail every dispatch in
    flight with a non-retryable `Protocol` error (one bad line failed every
    file on the worker) and then drop their real replies as orphans; it now
    fails no dispatch. A request-tagged error whose dispatch had already
    given up was handed to the sequential control op, so a late dispatch
    error could fail an `ensure_task` or capabilities probe; it now goes
    nowhere else.
  - Control replies are correlated with the op that asked. `capabilities`
    and `ensure_task` requests carry a control request id
    (`ControlRequestId`, `control-N`), which the worker puts on a failure
    line; the control slot holds a waiter typed by its op (and, for
    `ensure_task`, its `InferTask`), and a line answers it only when it
    names it. An `ensure_task(A)` that timed out could have its late
    success taken as the answer to the next `ensure_task(B)`, which then
    cached B as loaded for the worker's life, and its late failure could
    fail B. Task names are `InferTask` from the dispatcher to the wire
    (`ensure_task`, `EnsureTaskResponse.task`, the loaded-task sets,
    `WorkerWait::EnsureTask`); a sequential worker's reply about another
    task is a protocol error.
  - At EOF or a read error every waiting dispatch, and the waiting
    sequential op, is answered with `ProcessExited` through its own reply.
    The reader used to drop their senders, which stdio mapped to
    `ProcessExited` and TCP to a non-retryable `Protocol` error. A
    sequential TCP worker closing its connection is `ProcessExited` too; it
    was a terminal `Protocol` error.
  - Retiring one shared worker while the pool keeps running (its capability
    report was refused, or it is being replaced) answers the requests in
    flight on it with the retryable `WorkerError::WorkerRetired`
    (`WorkerCrash`). Every stop used to answer them `PoolShuttingDown`,
    which is terminal, so a refused capability report on a worker shared
    with other jobs failed their files. `PoolShuttingDown` is now only the
    pool's own shutdown: the stop reason (`Retirement`) is given by the
    caller.
  - A spawned shared worker is retired when one of its requests times out:
    the Python thread serving it may still hold one of the worker's slots,
    and enough such hangs made every later request time out. A daemon
    connection is kept (the daemon is not this server's to restart). The
    `WorkerError::Timeout` documentation said every timed-out worker was
    retired.
  - A TCP GPU daemon connection has the stdio worker's liveness check.
    Dispatch used to pick a registry daemon unconditionally, so after its
    stream closed every request waited out its full timeout (1800 s for audio
    tasks) and was retried on the same dead connection, and rediscovery never
    replaced the entry. A closed connection is now dropped at dispatch and
    replaced at discovery.
  - The sequential ops read a reported failure's kind: an `ensure_task`
    failure is `Bootstrap` (a refusal `RequestRefused`) on both transports,
    and a capabilities failure keeps its kind. The control channel carried
    text only, so stdio reported both as `Protocol` and TCP `ensure_task` as
    `Bootstrap`.
  - An op the protocol does not name, and a sequential reply that does not
    decode, answer the waiting op with a refusal; both used to be dropped,
    leaving the op on its timeout.
  - Replies are decoded from the parsed line without a deep copy, pending
    dispatches are keyed by the typed request id, and a second dispatch
    under an id already waiting is refused instead of orphaning the first.
    The unused shared GPU health probe is removed.
- A stdio worker keeps its protocol stream to itself: at startup it moves
  the pipe the server reads onto a private descriptor and points descriptor
  1 and `sys.stdout` at stderr (`claim_protocol_stdout`), so a library's
  print is logged instead of reaching the server. Every reader (sequential
  stdio, sequential TCP, shared GPU) applies one line rule (`WireLine`: a
  JSON object is a message, anything else is noise) and one limit of 8
  consecutive noise lines, and logs at most 200 characters of a refused
  line; the shared GPU reader took a JSON scalar as a message and skipped
  noise without limit, and the sequential readers took non-JSON text
  starting with `{` as a framing error. A run of noise reaching the limit
  is the retryable `WorkerError::OutputNoise` (the worker is retired). The
  unused pre-ready flag (`_handshake_complete`) is removed.
- A worker wait that runs past its limit is `WorkerError::Timeout`, naming
  what was awaited (`WorkerWait`) and the limit, and is classified
  `WorkerTimeout` by its type. Retry used to depend on the word "timeout"
  appearing in a `WorkerError::Protocol` message, checked in ten places;
  the GPU reader quotes worker and serde text into `Protocol` messages, so a
  protocol failure mentioning a field named like `timeout_s` was retried
  as a timeout. Health-probe timeouts are `Timeout` rather than
  `HealthCheckFailed`, and TCP connect timeouts `Timeout` rather than
  `Protocol` (both classified as before).
- Transport timeouts are `PositiveSeconds` end to end:
  `ExecuteRequestV2::transport_timeout` replaces
  `timeout_seconds_with_config` (a `u64` that lowered the operator's typed
  overrides) and the unused `timeout_seconds`; the four execute timers, the
  capability, health, infer, connect and checkout limits, the `ensure_task`
  default and the `batch_infer` limit (`batched_items_timeout`, shared with
  the V2 text tasks and speaker embedding) are typed. A wait whose limit
  lies past what the clock can represent waits without end (`Deadline`)
  instead of panicking on `Instant + Duration`. `server.yaml`'s
  `memory_gate_timeout_s` refuses `0` when read; it used to accept it and
  reject a job at once.
- `ExecuteResponseV2::server_composed_failure` and `WorkerElapsedV2` are
  removed. `ExecuteResponseV2::elapsed()` returns `NonNegativeSeconds`, and
  serializing a response can no longer fail. Earlier, a composed failure
  carried first an invented `0.0` seconds and then a `NotMeasured` value with
  no wire form.
- Worker elapsed times are always readings the worker took. Python-composed
  `execute_v2` refusals (payload kind not matching its task, task not wired)
  report receipt-to-reply time instead of `0.0`, and the test echo reports
  its real time instead of a fixed `0.001`. An `ensure_task`
  `already_loaded` answer reports the time of its check instead of `0.0`;
  the Rust `EnsureTaskResponse.elapsed_s` is now `NonNegativeSeconds` (was
  `f64`), so a negative or non-finite reading is refused. A Rust-side
  `ensure_task` cache hit no longer builds an `already_loaded_cached`
  response with an unmeasured `elapsed_s` of `0.0`; it returns without a
  response.

### Time types

- pyannoteAI diarization bounds are admitted through
  `AdmittedInterval::admit_positions`. Since 12aa8553 they were converted
  with `AudioPositionSeconds::nearest_millis`, which saturates, so an
  epoch-like bound (about 1.7e9 seconds) became a valid millisecond count
  instead of being refused as the conversion it replaced refused it; the
  admission's range limit applies again, and its inversion check replaces a
  hand-written one.
- `AdmittedInterval` deserializes only through `admit_millis` (`try_from` a
  raw pair). It derived `Deserialize`, so a stored or received interval
  could be negative, inverted or out of range, and `as_positions` turned a
  negative bound into a positive position.
- `AdmittedInterval`, `IntervalRefusal` and `IntervalBound` moved to
  `batchalign_types::interval` (re-exported where they were), so the
  speaker wire type can use them: `SpeakerSegmentV2` holds one
  `AdmittedInterval` (wire shape unchanged, `{"start_ms", "end_ms",
  "speaker"}`) instead of two `DurationMs` lengths, and a segment is refused
  when it is parsed. The four inversion checks that re-validated segments
  are gone (PyO3 speaker output, turn building, legacy replay, the speaker
  evidence cache), with `TurnsBuildError::InvertedSegment`; replay reports
  `InvalidTurnInterval`. IPC schemas regenerated.
- An HK provider projection element holds one `Option<AdmittedInterval>`
  instead of two independent `Option<AudioPositionSeconds>` fields; the wire
  `ts` / `end_ts` are written from it, both present or both null.
- (Moved from the book.) The ASR element endpoints were `AsrTimestampSecs`,
  an enum `{ Observed(f64), Absent }` local to `batchalign-transform`: a
  hand-written copy of `Option` with an unproven `f64` payload, because the
  crate did not yet depend on `batchalign-types`. They became
  `Option<AudioPositionSeconds>` on 2026-10-01, and the enum, its
  `From<Option<f64>>`, `PartialEq<f64>` and `Display` were removed.
- Seconds reach `AdmittedInterval` only as positions:
  `WordTiming::from_seconds(Option<f64>, Option<f64>)` is replaced by
  `WordTiming::from_positions(Option<AudioPositionSeconds>, ..)`,
  `AdmittedInterval::admit_seconds` is private behind the new
  `admit_positions`, and `as_seconds() -> (f64, f64)` is replaced by
  `as_positions()`. The one production caller no longer lowers proven
  positions to `f64` and back.
- pyannoteAI segment bounds are read as `AudioPositionSeconds`, so a
  non-finite or negative bound is refused when the segment is parsed; the
  hand-written check after the fact is gone.
- HK provider projection elements carry `ts` / `end_ts` as
  `Option<AudioPositionSeconds>` instead of `Option<f64>` (the wire shape is
  unchanged).

### File locks

- One cross-process lock primitive, `file_lock::HeldFileLock`, replaces the
  three hand-rolled ones (the handshake slot lock, the daemon start lock and
  the media cache slot lock). It creates a missing directory, so retiring a
  handshake in a state directory that does not exist finds nothing instead
  of failing to create the lock file.
- A lock is released by an explicit unlock when it is dropped. Released by
  closing alone, it stayed held while a child forked by any thread still
  shared the descriptor (until that child execs), so a waiter checking at
  once saw it held. This made `server_handshake::tests::
  a_publish_cannot_land_inside_a_removal` fail intermittently when another
  test spawned a process at that moment; reproduced deterministically by
  `file_lock::tests::dropping_releases_the_lock_while_a_forked_child_shares_
  the_descriptor`.
- The worker registry (`workers.json`) is read-modify-written under
  `workers.json.lock` by every writer. The Rust server wrote it without a
  lock, and discovery rewrote it from a snapshot taken before its health
  checks, so an entry a daemon registered meanwhile was lost; it now removes
  stale entries by identity under the lock. The Python writers locked the
  registry file they then replaced by rename, which excludes nobody; they
  now take the same lock file. Python writes the registry `0600` through a
  private temporary file, as the Rust writer does; it wrote `0644`, so the
  file's mode depended on which writer wrote last.

### L2 (`@s`) morphotag

- Merged `@s` spans splice in transcript order. The dispatcher merges spans
  language by language from a hash map, and each splice reads the `%gra`
  the previous ones left, so the order could vary between runs on a line
  with spans in two languages. A secondary worker that answers fewer spans
  than were sent is now reported.
- A phrasal particle is a `compound:prt` word whose head the secondary
  tagged VERB. With the phrasal context now always present, the particle
  rule would otherwise have rewritten `out` in the compound noun `time@s
  out@s` (Stanza: `out` `compound:prt` under the NOUN `time`) to PART; it
  keeps the secondary's ADP.
- Inside a multi-word `@s` span the secondary model's relations stand, and
  the span's one external relation comes from its attachment source (the
  span word whose primary head lies outside the span), checked against the
  category of the secondary root that carries it. A per-word correction used
  to rewrite the secondary root's relation from another word's primary
  view: `we talked about los@s:spa niños@s:spa .` gave `niños` `NMOD` where
  `los`'s `obl` belongs. The merge now produces one merged span
  (`MergedL2Span`) from one planned span, and the splice consumes merged
  spans; it no longer regroups per-word results. An utterance-root
  attachment has no relation field, so `head=0` with a non-`ROOT` relation
  cannot be planned. The unused span grouping `group_l2_spans` (and
  `L2Span`) and `apply_l2_fallback` are removed.
- An `@s` word's category is always one a model assigned (decided
  2026-10-01). The secondary model owns category, lemma and features; the
  primary owns the attachment, and where its relation contradicts the
  secondary's category the relation is corrected, never the category. The
  merge used to pick a category from the primary's relation (its most likely
  POS), turn a copula predicate's VERB into ADJ, fall back to the primary's
  own guess, and default to NOUN, while keeping the secondary's lemma and
  features: `je dis ja@s:nld` gave `noun|ja` though both models said INTJ,
  and `it's@s working@s` gave `noun|work-Part-Pres`. Those rules are gone;
  a phrasal particle (`compound:prt`) is still written PART.
- Phrasal verbs inside an `@s` span are recognised (`wake@s up@s` gives
  `part|up` with `COMPOUND-PRT`). The secondary sentence context reached the
  merge only when the secondary's UD word count equalled its mapped `%mor`
  count; the secondary is dispatched with Stanza owning tokenization, so the
  terminator row made them differ and the context was withheld for every
  span, without a report. The context is now each span word's place in the
  span's alignment, always present; a secondary analysis that does not
  align or map is an `L2MergeError` for the span, reported, and its words
  stay `L2|xxx`.
- An `@s` word's primary structure is read from its own UD word. Extraction
  indexed the UD word list (which holds multi-word-token range rows) by CHAT
  word index, so after a contraction it read another word: in `avui anem al
  cole@s per jugar .` `cole` got the `CASE` of the `a` in `al` instead of
  Stanza's `obl:arg`. The planner compared a CHAT index plus one with a UD
  head id when deciding whether a span word attaches inside its span, the
  same confusion. Both now read one typed alignment (`UdAlignment`), built
  once per utterance. An utterance whose analysis does not align is reported
  (`L2Extraction::unaligned`) and its `@s` words stay `L2|xxx`; a missing UD
  word used to get an invented `dep` relation at the utterance root.

### Morphosyntax mapping

- Chatter dependencies are pinned to the published `v0.28.0` tag. The prepared
  L2 migration uses the same public release rather than a local path override.
- Italian multiword-token collapses consume a producer-created `WalkedUdToken`
  view, retaining the representative selected by `UdTokens::walk`. A non-first
  external representative keeps its attachment and dependents. Independent
  attachment selection, the first-component assumption and the guessed-root
  fallback are removed; malformed ranges remain rejected by the original walk.

- The L2 splice consumes Chatter's checked `SplicedBlock`, `SpanRoot` and
  `HostRedirects`. A growing secondary span preserves a later governor's
  identity, and host dependents of the primary attachment source follow
  the secondary block's admitted root (`root_chunk`). Pre-splice host indices
  are translated by Chatter, not interpreted as post-splice indices.
  Manual root rewrites, duplicate-root demotion, anchor-cycle traversal and
  the independent repair-root return are removed. The caller explicitly
  requests `DEP` attachment when a primary root survives outside the span;
  refused admission retains the placeholders without changing either tier.
  `gra_upgraded` distinguishes planned external corrections from fallback
  attachments. The growing-span regression is enabled, with additional
  farther-governor and primary-source-dependent cases.

- `--retokenize` with a special form keeps the form on its own word. The form
  was sent to the model as `xbxxx`, the text mapping between the CHAT words
  and the model's tokens failed, and the length-spread fallback placed items:
  in `I don't like gumma@c .` the main tier lost `@c` and the form's
  analysis could land on a neighbour. The form's token is now written back
  as the CHAT word, so the mapping holds; an utterance whose special-form or
  `@s` words would be placed by a length-spread mapping is reported instead.
  The rebuild reuses the mapping injection placed items by
  (`retokenize_utterance` takes it), instead of building it a second and
  third time.
- A UD sentence is walked once (`UdTokens::walk`), and both the `%mor`
  mapper and the CHAT-word alignment consume that walk. They used to walk it
  separately by rules that differed: the mapper wrote a multi-word token's
  range row as a word of its own when the components after it were missing,
  and let a repeated UD id overwrite the first in its head table, where the
  alignment refused both. A malformed sentence is now one error for both
  (`MappingError::Sentence`), reported as a mapping failure for the
  utterance. The L2 merge maps the alignment's own walk, so its check that
  the mapper and the alignment agree on the word count is gone.
  `map_ud_sentence_with_overrides`, `TerminatorPolicy` and the public
  `build_gra_and_validate` are removed: the mapper always appends the
  terminator relation, which was the only policy used.
- A morphotag batch item holds one record per word (`BatchWord`: the text
  the model receives and the word's `WordRole`: analysed, a special form, or
  code-switched) instead of a word list and a parallel list of
  `(form type, language)` pairs. The two lists could disagree in length,
  which L2 extraction checked at run time (`L2ExtractError::FormsWordsMismatch`,
  removed); a special form's placeholder text now follows from its role in
  the record's constructor rather than from a second walk over both lists.
  The worker receives the same JSON, and the published schema is unchanged.

- A special form or code-switched word after a contraction is relabelled at
  its own `%gra` chunk. The relabel paired `%mor` items with `%gra` relations
  by position, but a contraction is one item of two chunks: in `it's
  gumma@c .` it rewrote the clitic `'s` from `COP` to `DEP`. Words are now
  placed through the walk's alignment to the CHAT words (preserve mode) or
  the text mapping that rebuilds the main tier (retokenize mode), and each
  item's relation is found at its own first chunk.
- `@s` positions are read from the analysis the `%mor` mapping reads. L2
  extraction ran on the raw primary analysis while injection mapped it after
  the grammatical invariants rewrote it, so a word the contraction rule
  moved was deferred with its old structure (`you hafta put@s:spa .`: `put`
  deferred as the utterance root, its `%gra` an `XCOMP` under `have`).
  Extraction is now a step of injection (`InjectionResult::l2`), over the
  same rewritten walk; `extract_l2_deferred_positions` is removed. An
  utterance that is not injected defers nothing (its decision record reports
  it). `L2DeferredPosition` and `PrimaryStructuralInfo` have private fields
  and are built only from an aligned word.
- With `--retokenize`, an `@s` word after a contraction is placed by the
  `%mor` item it was written as. The main tier is rebuilt with one item per
  model word, but special forms and `@s` positions were placed by CHAT word
  index: in `gonna see camino@s:spa .` (`gonna` = `gon` + `na`) `see` was
  written `L2|xxx`, `camino` kept the primary's analysis, and the L2 splice
  targeted `see`. A position now names its own item and its head word's
  item; an `@s` word that became several items, or whose placement by text
  and by analysis disagree, is reported and stays `L2|xxx`.
- `--retokenize` rebuilds the main tier from the model's own tokens (the
  analysis as returned, before the grammatical invariants), and an analysis
  whose item count differs from them is a reported retokenization failure
  that leaves the utterance as it was (`SurfaceItems::pair`). The rebuild
  had moved to the rewritten analysis's words, so the English contraction
  table's expansions reached the transcript: `I hafta go .` was rewritten
  `I have to go .` (also `gimme`, `lemme`, `dunno` and the rest of the
  table).
- `--retokenize` with an MWT lexicon (`--lexicon`) writes a valid `%gra`.
  Expanding a token repeated its relation under the same index and dropped
  the terminator's relation (a truncating zip of tokens, items and
  relations), so with any non-empty lexicon every utterance failed
  injection. Each later piece of an expansion is now its own item attached
  to the first as `FIXED`, every index and head is renumbered, and the
  terminator's relation is kept (`SurfaceItems::expand`).

- L2 splice: `gra_upgraded` counts corrected external relations actually
  written under the planned host governor, not spans carrying a correction.
  Utterance-root replacements and generic primary-root `DEP` fallbacks are
  not counted. A correction that was not written was still counted before.
- L2 splice: a host anchor that cannot be read (the attachment source has no
  `%gra` relation) falls the span back to `L2|xxx` with a
  `host_anchor_unreadable` warning; it used to be read as "no anchor", and
  the span root took the replaced block's old head. A host governor that is
  the utterance root is now that: the span root becomes the root (it used to
  be "no anchor" too). Host indices are `SemanticWordIndex1`; admitted
  block-relative roots and redirect targets are `BlockChunk`.

- L2: a span's external relation is held once. A merged span carried the
  attachment's relation and, beside it, the correction as
  `corrected_deprel`, and a plan's `host_deprel` copied the source's primary
  relation. The attachment is now typed by stage: a plan's
  (`L2Attachment<AsPlanned>`) has the source's primary relation and no
  field for another; a merged span's (`L2Attachment<ExternalRelation>`)
  says `Primary` or `Corrected(relation)`.

- L2: repairing a span's relations tracks one root through every pass;
  final tree admission and the root accessor belong to `SplicedBlock`, not
  an independent repair-root return. Two separate cycles in a
  secondary analysis with no root got a root each: the existing root was
  read once, before the cycle pass, so its first promotion did not count for
  the next. A `head=0` row with a label other than `ROOT` is now counted as
  a root too (the first is labelled `ROOT`, the rest attach to it).

- A mapped `%mor` item and its chunks' provenance are one value with
  private fields. `MappedItem` is built a chunk at a time (`word`, then
  `with_post_clitic`), each chunk with its `ChunkProvenance`, so the
  GRA builder's chunk-count check and `MappingError::ChunkCountMismatch`
  (with the splice's `secondary_chunk_count_mismatch` category, which
  nothing could produce) are gone. `ChunkProvenance` carries typed UD ids
  (`UdWordId`) and is built by one constructor per kind of chunk; the
  Italian overrides build their clitic chunks through the same item
  builder instead of a second provenance loop beside the `%mor` one.

- A UD word's head is typed: `UdHead::Root` or `UdHead::Word(UdWordId)`,
  converted from the analysis's `HEAD = 0` once, where the analysis is
  admitted. The English contraction expansion and the copula-progressive
  rescue move heads as typed ids, and `UdWord::synthetic` takes the word's
  real id instead of giving every synthesized word id 0; the Italian
  overrides' readings (`UdWord::curated_reading`) take the id, head and
  relation of the word they read, where their clitics had head 0.

- A collected utterance is a record, `CollectedUtterance`, with private
  fields (its line as a `LineIdx`, its ordinal, the item sent, its words),
  built only by `collect_payloads`, instead of a four-slot tuple. The
  inference functions take any slice of items (`AsRef<MorphosyntaxBatchItem>`),
  so the secondary `@s` dispatch sends bare items: it no longer fills a
  line index and an ordinal with placeholder zeros and an empty word list.

### Tests

- `scripts/ba2-morphotag-divergence.sh` reports a `chatter validate` that
  failed without a validation error (a crash, an unreadable file) as not
  validated, with its exit status; it discarded the status and labelled such
  a fixture valid CHAT.
- `scripts/ba2-morphotag-divergence.sh` writes, for each fixture BA3
  produced no output for, its first `chatter validate` error. Three parity
  fixtures are invalid CHAT and are kept so, since BA2's references were
  generated from them as they are (`eng_bilingual`, `eng_complex_tiers`,
  `spa_clinical`); the script header and the fixture provenance say so.
- The ML golden morphotag tests use the shared golden helpers
  (`parse_output`, `has_mor_tier`, `find_mor_line_for`) instead of a second
  copy in `morphotag/helpers.rs`. The Python golden directory's unread
  `golden_dir` fixture (`conftest.py`), `fixtures/` (eight `.cha` files)
  and `manifest.json` are removed; `ba2_reference/` stays.
- A Stanza bump now re-runs the Rust ML golden morphotag suite
  (`BUMP_PROCEDURE` step 2, exact command), whose snapshots drifted for five
  months because nothing ran them when the model or the code changed.
- The sixteen Python `batchalign/tests/golden/expected/*.expected` files,
  which no test read, and the unused `update_golden` fixture with its
  `--update-golden` option are removed (decided 2026-10-01). Three book
  pages that named `batchalign/tests/golden/` as the golden baselines now
  name the ML golden snapshots.
- Morphotag is compared with Batchalign 2 by a report, not a test (decided
  2026-10-01): `scripts/ba2-morphotag-divergence.sh OUTPUT_DIR` writes a
  `compare-runs morphotag` divergence report against BA2's January
  references. The nine `ml_golden::morphotag::parity` tests, which asserted
  exact line equality with BA2 output that BA3 departs from on purpose, are
  removed; the reference data stays. Other commands' parity tests are
  unchanged.

### Documentation

- The Italian language page named the deleted `map_ud_word_to_mor`; it is
  `map_ud_word`. The stanza-limitations `hafta` example showed fills the
  renderer no longer writes (`pron|you-Prs-Nom-S2`, `verb|put-Inf-S`); it now
  shows what the mitigation test expects (`pron|you-Prs-Nom-2`,
  `verb|put-Inf`, `det|that-Dem-Sing`, `noun|one`).

### File modes

- Atomic writes state their audience. Daemon state, server handshakes,
  `workers.json` and debug artifacts are `0600` on purpose (per-user state;
  `fs::write` gave them `0644` before they moved to the atomic writer, which
  made them `0600` as a side effect). `compare-runs` reports and `eval
  utr-alignment` evidence now get `std::fs::write`'s mode (`0666` less the
  umask); their temporary-file writers, before and after the move to the
  shared atomic writer, created them `0600`.

### `%mor` rendering

- A pronoun's number is written as its initial, as on every other category:
  `Dual` is `D`. It was written `S` for any non-plural value, a singular
  nobody assigned.
- Every feature value on a UD word says where it came from, as a typed field
  of the value: the analysis that returned the word, or one of our curated
  tables. Every table that writes features is covered: the English contraction
  expansions, the Italian reconcilers (mis-split overrides, compound
  imperatives and their clitics, component rewrites), the copula-progressive
  rescue and the lexicon category constraint. The table values used to be
  indistinguishable from the analysis's for all but the contractions, whose
  record lived in UD's MISC column, where a value the worker sent could claim
  it and a later rewrite of the word's features left it behind. FEATS is
  private to the word and replaced whole by each writer; MISC carries no
  provenance; a table's FEATS string is checked at compile time; the three
  FEATS splitters are one. Rendering is unchanged.
- The renderer no longer writes invented `%mor` values (removed with
  authorization). It wrote eleven values the analysis did not contain, by the
  convention BA3 inherited from Batchalign 2. They were fabrications (an
  adjective has no person, an English noun no case, `king` is not a gerund), and
  each was a named `LegacyFill` variant, counted per language in
  `InjectionResult::legacy_fills`; the type, the count and their plumbing are
  gone with them.

  | Word | No longer written | Example, before and after |
  |------|-------------------|---------------------------|
  | verb, aux without `VerbForm` | `Inf` | `verb\|说-Inf-S` to `verb\|说` |
  | verb, aux without `Number` | number `S` | `verb\|walk-Inf-Past-S` to `verb\|walk-Past` |
  | pronoun without `PronType` | `Int` | `pron\|there-Int-S1` to `pron\|there` |
  | pronoun without `Number` / `Person` | `S` / `1` | `pron\|you-Prs-Nom-S2` to `pron\|you-Prs-Nom-2` |
  | determiner without `Definite` | `Def` | `det\|this-Def` to `det\|this` |
  | French determiner without `Gender`, unless plural | `Masc` | `det\|ce-Masc-Def` to `det\|ce` |
  | adjective without `Number` / `Person` | `S` / `1` | `adj\|forte-S1` to `adj\|forte` |
  | noun without `Case` whose relation is `obj` | `Acc` | `noun\|dish-Acc` to `noun\|dish` |
  | English noun spelled `-ing` | `Ger` | `noun\|king-Ger` to `noun\|king` |

  On Stanza 1.15.0 analyses of an 80-file, ten-language sample, rendered before
  and after by the deploy-free harness (see the testing page), the only `%mor`
  changes were these removals: 3,240 `%mor` lines, every changed item equal to
  its old form with fills removed, and the removals per kind exactly the counts
  the old tally recorded (12,032 in all: a verb's number 3,463, `Inf` 2,072,
  a pronoun's number 1,300 and person 1,073, an adjective's person 1,117 and
  number 594, an object's `Acc` 1,083, `Def` 762, `Int` 546, `Ger` 16, `Masc`
  6). No main tier or `%gra` changed.

### Worker configuration

- Retiring this server's TCP daemons kills and removes them in one pass
  under the registry's lock; the kill read an unlocked snapshot, so a
  daemon registered between the read and the removal was dropped from the
  registry without being killed. The registry and handshake locks are taken
  off the async runtime (`spawn_blocking`) at shutdown, at publish and
  retire, and in `worker stop`. Removing a handshake from a state directory
  that does not exist creates nothing (it created the directory and a lock
  file). The Python registry writer fsyncs the directory after its replace,
  as the Rust writer does, and its Windows unlock releases the lock
  (`LK_UNLCK` at byte 0; it was `LK_NBLCK`, a second lock).
- Operator timeouts stay `PositiveSeconds` from `server.yaml` to the timer:
  `PoolConfig.health_check_interval_s` and `ready_timeout_s`,
  `WorkerConfig.ready_timeout_s`, `WorkerError::ReadyTimeout::timeout_s` and
  every `ensure_task` timeout were `u64`, lowered at the first hop, and the
  memory guard re-checked the ready timeout with `.max(1)`.
  `PoolConfig::effective_ensure_task_timeout_s` is now
  `effective_ensure_task_timeout` and returns `PositiveSeconds`.

### Server handshake

- A stopping server can no longer delete the handshake a replacement server
  has just published. Retiring read the record and then deleted it with no
  lock, so a publish landing between the two was lost. Publishing and the
  read-then-delete now run under a per-slot lock file (`server.pid.lock`,
  `sidecar-server.pid.lock`) held for one file operation.

## Earlier changes (moved from the book)

### Book history moved on 2026-10-01

- `LeaseRecord`'s fields were public, so a lease whose expiry was not after
  its heartbeat was constructible anywhere, until it got private fields and
  the `new` / `taken` / `renew` constructors.
- `Submitter` replaced two `String` fields that used `""` for absence until
  the API projection.
- `TaskTimeoutOverrides` replaced a pair of `u64` parameters, `0` meaning
  default, carried separately through `PoolConfig`, `WorkerConfig`,
  `TcpWorkerInfo`, the registry walker and the shared GPU worker.
- Once the ASR and FA wire types carried `AudioPositionSeconds`, the checks
  their consumers repeated (`replay`, the Rev evidence admission, the PyO3 FA
  parser, the protocol matrix) were deleted.
- `MachineTime` replaced `UnixTimestamp(f64)`, which admitted `NaN` and any
  number and made every consumer convert.
- `NonNegativeSeconds` restated its OpenAPI schema by hand beside the macro's
  JSON Schema until the `validated_numeric!` schema clause generated both.
- The atomic writes each renamed a fixed `<target>.tmp` of their own, which
  two racing writers could truncate under each other, until they moved to
  `atomic_file::write_atomically`.
- A stop printed "No server process found" for a dead, absent and unreadable
  record alike before `StopOutcome`, and the daemon path loaded
  `server.yaml` up to four times per invocation before it was handed one
  `ServerConfig`.
- The `WorkerError::Bootstrap` variant was added on 2026-05-06; every other
  variant kept its retryability then. Before it, a deterministic bootstrap
  failure was `WorkerCrash`, retryable, and the retry loop ran it three
  times.

### L2 (`@s`) morphotag history

- L2 morphotag shipped behind `--experimental-l2-morphotag` with the
  POS priority chain and contiguous span dispatch, and was made the default
  (`--no-l2-morphotag` to opt out) after an evaluation across 19 language
  pairs (17 at 100% dispatch, `cym,eng` at 99.8%, `eng,yue,zho` at 99.9%,
  99.96% aggregate).
- Contraction expansion for `@s` words: English added to `MWT_LANGS`,
  range parents filtered from the retokenize token vector,
  `map_ud_sentence_expanded` for the retokenize path, and the secondary
  dispatch switched to `retokenize=true`, so `it's@s` became
  `pron|it~aux|be`. A later Python regression in `_tokenizer_realign.py`
  stripped Stanza's MWT hint tuples before realignment; `_realign_sentence`
  and `_conform` now overlay them.
- Phrasal verbs: the secondary UD sentence was threaded into the merge
  (Priority 0), so `give@s up@s` went from `adv|give adp|up` to
  `verb|give part|up` with `COMPOUND-PRT`. (Until 2026-10-01 a length
  guard withheld that context whenever the secondary sentence kept its
  terminator row; see Unreleased.)
- Primary `--retokenize` for non-CJK languages: `gonna eat cookies .`
  became `gon na eat cookies .` with one `%mor` item per component.

- `JobSourceContext::submitter` replaced two `String` fields in which `""`
  meant "no submitter" above the database, which conflict detection, the
  store and recovery all carried as a value.
- ASR word timing: a negative bound with a later end used to reach a word as
  a negative millisecond time, and a value beyond `i64` saturated into a
  plausible one, before `AdmittedInterval` owned the conversion.
- `PositiveSeconds` replaced `u64` config fields in which `0` meant "the
  built-in default" (`audio_task_timeout_s`, `analysis_task_timeout_s`,
  `ensure_task_timeout_s`, `worker_health_interval_s`,
  `worker_ready_timeout_s`, and `PoolConfig`'s checkout wait).
- `FilePhase` replaced a file status beside five settable `Option` fields
  that five mutators reset by hand, which let a file awaiting a retry report
  `Processing` with a `finished_at`, and so a duration, while still running.
- `NonNegativeSeconds` replaced `DurationSeconds`, an unvalidated `f64`
  newtype with a public field and a `Default` of zero that carried lengths
  and ASR token positions alike; `InferResponse.elapsed_s` read a missing
  field as zero through that `Default`.
