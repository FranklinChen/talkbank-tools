# utseg: Developer Reference

**Status:** Current
**Last updated:** 2026-09-16 08:18 EDT

Implementation guide for the `utseg` command. For user-facing documentation,
see [User Guide: utseg](../../user-guide/commands/utseg.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `UtsegArgs` | lang, num-speakers |
| Catalog entry | `crates/batchalign/src/recipe_runner/catalog.rs` | the `CatalogEntry` for `utseg` |
| Stage recipe | `crates/batchalign/src/recipe_runner/recipes.rs` | `UTSEG_RECIPE` |
| Utseg orchestration | `crates/batchalign/src/utseg.rs` | The cross-file batch pipeline (standalone jobs) and the per-file pipeline transcribe reaches through `process_utseg_with_evidence`; typed worker-result admission and provenance |
| Worker IPC | `batchalign/inference/utseg.py` | Returns direct model assignments with evidence, or Stanza trees |
| Python model evidence | `batchalign/models/utterance/evidence.py` | Closed actions, fixed-point probability, omission/bypass states |
| Canonical IPC evidence | `crates/batchalign-types/src/worker_v2/utseg_evidence.rs` | Rust wire enums and validated probability newtype |
| Evidence artifacts | `crates/batchalign/src/utseg_evidence.rs` | Versioned pre/post-CHAT transcribe traces, atomic sink, and the admission that reads one back |
| Evidence replay | `crates/batchalign/src/cli/eval_cmd/utseg_replay.rs` | `eval utseg-replay`: reapplies a retained sidecar and compares the result with the retained output |
| Boundary application | `crates/batchalign-transform/src/utseg.rs` | Maps admitted assignments back to typed CHAT structure |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## Caching behavior

Text NLP tasks (`utseg`, `translate`, `morphotag`) do not use the utterance cache.
Boundaries are computed from word sequence, language, exact model revision, and
current postprocessing during each inference run; no per-utterance result cache
exists. Server mode still avoids repeated model-loading startup cost by keeping
the worker warm. Transcribe debug evidence can be replayed for policy research
without rerunning the boundary model, but it is not a production result cache.

---

## Worker IPC: utseg task

Rust freezes the text batch in a prepared artifact, then sends
`execute_v2(task="utseg")` with the language and explicit Stanza-fallback
authorization. Each result item has exactly one success representation:

- `assignments`: one group ID per request word. A boundary-model result also
  carries `boundary_model_evidence`, including model identity and one typed
  evidence state per word.
- `trees`: raw constituency trees from the explicitly authorized Stanza
  fallback. Rust computes assignments from the trees.
- `error`: a per-item failure with no success payload.

`AdmittedUtsegPrediction` rejects error/success mixtures, assignments plus
trees, evidence without assignments, empty model identity, and any assignment
or evidence length that differs from the request words. Its variants preserve
boundary-model, unobserved direct-assignment, and constituency sources, and the
batch path keeps them all the way to the stamp, which is how a file's `engine=`
names the boundary model that segmented it.

`admit_prediction` is the one constructor of that type. `admit_worker_item`
classifies a live worker result's mutually exclusive payloads into a
`UtsegPredictionOrigin` and hands it over; evidence read back from disk takes
the same route. One constructor means one copy of the parallel-vector and
consistency checks, so a retained sidecar is admitted because it was checked,
not because it is on disk.

A locally rederived decision carries a `LocalUtsegDecisionReceipt`, and because
a receipt can be deserialized from an artifact, `with_local_decision` treats it
as a claim rather than a label: `check_explains` recomputes through the same
owner that performed the reapplication, requiring that the evidence declare the
policy the receipt names (reapplication stamps it there), that the worker's own
declared policy reproduce the worker assignments recorded, and that replaying
the local policy reproduce both the applicable assignments and the exact
suppressions claimed.

Because admission already refuses a non-parallel prediction, the batch
application has no length check and no "keeping original" branch. That branch
was unreachable, and had it ever run it would have produced silently
unsegmented output where admission produces a failure.

`engine=` carries no placeholder. A boundary model is written
`<model id>@<revision>` and never by its id alone; the Stanza fallback is
`stanza-constituency`; and a worker that returns assignments without naming
their source contributes no name at all, so a file with no named source gets no
stamp and the run records `NoStampReason::SourceNotNamed` on that file. The
invented names this replaced (`unobserved-worker`, `<id>@unrecorded-revision`)
were exactly what W0 and W1 removed elsewhere.

The revision is a required part of a model's identity, so the id-only form has
no representation rather than merely no callers: `UtsegBoundaryModelEvidenceV2`
carries a `HubCommitV2`, not an `Option<String>`, and the renderer has no branch
that could emit an unqualified id. That is a consequence of the load, not a rule
imposed on it. The boundary model is resolved from a pinned snapshot
(`model_manifest::UTSEG_BOUNDARY_MODELS`, sent to the worker under
`PINNED_UTSEG_MODEL_KEY`), and the commit is read off the directory that
actually exists on disk, so a worker that cannot say which revision it loaded
refuses instead of reporting none. Requiring the revision BEFORE pinning the
load would have refused every run rather than fixed anything.

---

## The segmenter route, decided at planning time

`crates/batchalign/src/utseg_route.rs` owns one question: which segmenter a
language gets. `UtsegRoute::resolve(lang, fallback_policy)` answers
`BoundaryModel`, `StanzaFallback`, or the typed refusal `UtsegUnavailable`.

Three things about its shape are deliberate:

- **One table.** `model_manifest::UTSEG_BOUNDARY_MODELS` is the only statement
  of which languages have a TalkBank boundary model, and it is the same table
  that names the model and pins its commit. There were three: a `matches!` in
  `pipeline/transcribe.rs`, a `BOUNDARY_MODEL_LANGUAGES` list in
  `utseg_route.rs`, and the key set of `_RESOLVER["utterance"]` in
  `batchalign/models/resolve.py`, which the worker consulted to decide whether
  to refuse. Availability is now a CONSEQUENCE of the pin
  (`has_boundary_model` asks the manifest), so a language this build claims to
  segment and a language it can name a model for are the same set by
  construction. The Python resolver no longer carries an `utterance` family at
  all: an id must be known before a load in order to pin its revision, so Rust
  owns it and sends it with the spawn.
- **"Refused" is the error arm, not a variant.** Both variants of `UtsegRoute`
  are runnable. A `Refused` variant would have been a value every consumer had
  to remember to reject, which is the shape that let the old refusal be
  skipped; as an `Err` it cannot be stored in a plan or handed to a worker.
- **It reads the language and the policy only.** The refusal it replaces was
  computed inside the worker from the payload, below a short-circuit that
  answered single-word items trivially, so a batch whose items were all one
  word long bypassed it and reported success for a language with no segmenter.

Resolution happens where the answer is first knowable:
`TranscribeDispatchPlan::from_job` refuses before any ASR is dispatched, and
`runner::routing` refuses a standalone `utseg` job before dispatch. Under
`--lang auto` the language is not known until ASR returns, so that case alone
resolves the same route inside the pipeline instead.

The worker keeps its own `UtsegModelNotFoundError`. It is NOT the same
predicate and was deliberately left in place: it fires when this worker
process has no boundary model LOADED, which can happen for a language that
has one, so it stays as a worker-local backstop rather than the decision that
fails a job after ASR has run.

## Stanza constituency availability

About 11 languages have Stanza constituency models. A language without a
configured TalkBank boundary model is refused by default; the operator must
pass `--utseg-fallback-stanza`. The available processors are queried at worker
startup via `batchalign/worker/_stanza_capabilities.py`, never hardcoded.

## Evidence retention scope

The transcribe pipeline can retain exact boundary evidence with
`--debug-dir`: pre-CHAT and post-CHAT phases get separate files. The standalone
`utseg` command currently admits the same typed worker result but deliberately
projects it to assignments without writing those transcribe sidecars. Do not
claim that a standalone run retained evidence unless a future command-specific
artifact surface explicitly does so.

## Replaying retained evidence

`batchalign3 eval utseg-replay` reapplies a retained sidecar and checks that it
still produces the document the run wrote. `post-chat` consumes the
`_pre_utseg.cha` dump, the post-CHAT sidecar and the `_post_utseg.cha` dump;
`pre-asr` consumes the retained `*_asr_response.json`, the pre-CHAT sidecar and
the `_post_asr.cha` dump, and rebuilds CHAT through transcribe's own functions
(`prepare_asr_chunks`, `build_prechat_utseg_items`, `apply_prechat_assignments`)
so the replay cannot drift from the code it checks. Each mode names the phase it
reproduces when it admits the artifact, so a pass cannot be replayed against the
other's evidence.

Chunk preparation is one implementation, not two: `prepare_asr_chunks_with_snapshot`
delegates to the transform's `prepare_asr_chunks` when there is no trace to
fill, and the traced path is held to the same answer by
`snapshot_and_plain_preparation_agree`, which is the test that would catch its
digit fast path diverging on number expansion.

The input gate is production's: the document is parsed leniently and then judged
by `validate_to_level` with its parse errors in hand, exactly as the utseg
pipeline does, so the replay refuses what the run would have refused. The
pre-ASR pass additionally refuses a retained output whose `@Languages` names
anything but the single language the evidence records, or that carries a
`[- code]` precode, because those are the marks of a `--lang auto` run's
per-file and per-utterance detection, which this pass does not perform.

Binding is what makes the comparison meaningful: the requests the current build
collects must match the retained items one for one, in count, transcript
position, words and text. Both passes then compare on one basis, the AST of the
serialized CHAT text (`comparison_basis`), because text is what a run writes and
what every later stage sees; doing it in both keeps a serialization-only defect
visible in both. The comparison sets aside the comments a run generates (the
`[fc-ba3 ...]` stamp and the unchecked-ASR warning), recognized through
`provenance::recognize_generated_comment`, the same codec `extract_provenance`
reads with. A stamp's timestamp can never match by equality, so comparing it
would report every replay as a difference.

The verdict is a typed outcome: reproduced, or a difference naming the
comparable line counts and where they first differ. A difference exits 1, a
refused input exits 2. See [the user guide](../../user-guide/commands/eval.md)
for the flags and the admission rules.

## Replaying adjacency policies

`scripts/probe_utterance_boundary_policy.py` compares the current decoder
policy with a boundary-only alternative over retained
`*_asr_response.json` artifacts. It makes no ASR request. The local model is
loaded once, raw evidence is captured once per source monologue, and both
policies are applied to that identical evidence.

```bash
uv run python scripts/probe_utterance_boundary_policy.py \
  <retained-asr-directory> \
  <output-report.json>
```

The probe validates the complete retained input schema, records SHA-256 for
every input, refuses model-identity or evidence-length drift, and atomically
publishes a versioned report. Each assignment-changing case includes lexical
context, fixed-point boundary probability, and a typed known-or-missing
interword timing. The report is experimental evidence, not a production-policy
switch or an accuracy verdict. Candidate promotion requires a human-linked,
controlled comparison.

---

## Pre-validation gate

`utseg` requires CHAT Level 1 (parseable + valid headers). Gate in
`crates/batchalign/src/utseg.rs`. Implemented via
`validate_to_level(chat, ValidationLevel::StructurallyComplete)`.

---

## Testing

```bash
make test
cargo test -p batchalign utseg::
cargo test -p batchalign utseg_evidence::
uv run pytest -q batchalign/tests/models/test_bert_utterance_sliding_window.py
uv run pytest -q batchalign/tests/models/test_utterance_boundary_policy.py
uv run pytest -q batchalign/tests/models/test_utterance_policy_probe.py
uv run pytest -q batchalign/tests/pipelines/utterance/test_utseg_inference.py
# ML golden tests, only on Fleet/Large-tier hosts
cargo test -p batchalign --features ml-golden --test ml_golden utseg::golden
```

---

## Related developer documentation

- [Command Flowcharts: utseg](../../architecture/command-flowcharts.md#utseg)
- [Utterance Segmentation](../../reference/utterance-segmentation.md)
- [Stanza Capability Registry](../../architecture/stanza-capability-registry.md)
