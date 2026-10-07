# Command Contracts: Input Admission and Output Guarantees

**Status:** Current policy; command-admission migration in progress
**Last updated:** 2026-10-06 23:58 EDT

## One validity owner

Chatter defines CHAT validity. Batchalign defines what a command will replace.
Every retained part of an input must be valid before inference. A command may
accept a defective generated tier only when its actual selected plan discards
and regenerates that tier. Being uninterested in a field is not an exemption:
preserved headers, timing and dependent tiers still belong to the output.

Legacy `ValidityLevel` checks are limited workflow checks, not certificates of
full CHAT validity. In particular, `StructurallyComplete` and `MainTierValid`
do not run every main-tier, header, timing or alignment rule. Raising a level
does not implement complete admission.

## Required command policy

| Command/plan | Permitted replacement exemption | Retained input |
| --- | --- | --- |
| `morphotag`, replacing analysis | `%mor` and `%gra` actually replaced by that plan | Everything else must be valid |
| `morphotag`, keeping existing analysis | None for preserved `%mor`/`%gra` | Everything preserved must be valid |
| `align`, regenerating word timing | `%wor` actually replaced by that plan | Everything else must be valid |
| `align`, keeping `%wor` or skipping alignment | None for preserved `%wor` | Everything preserved must be valid |
| `utseg` | No blanket dependent-tier exemption | Complete valid CHAT before payload collection |
| `translate` | No blanket dependent-tier exemption | Complete valid CHAT before payload collection |
| `coref`, including non-English pass-through | No blanket dependent-tier exemption | Complete valid CHAT before inference or pass-through |
| `compare` / `benchmark`, reference transcript | None | Complete valid reference before model inference |
| `align`, incremental prior | None | Complete valid prior before timing can be copied |

An exemption is bound to the file and its selected plan, not merely the command
name. A no-op, `dummy`, CA or `NoAlign` path does not gain authority to write
invalid CHAT because another path could replace its defective tier. Invalid
headers, main tiers and retraces are never regeneration exemptions.

Recovery remains useful for diagnostics and inspection. A recovered model is
not an admitted inference input. Admit through the actual parser and validator;
do not strip source lines, reparse a cleaned substitute or add a parallel parser.

## Current enforcement

The shared text pipeline in `crates/batchalign/src/pipeline/text_infer.rs`
uses Chatter's source-bound admission with model rules and dependent-tier
alignment enabled. This covers the single-file and cross-file `utseg` paths
and the cross-file `translate` path. Batch inputs carry their actual transcript
name for filename-dependent checks. String-only entry points are explicitly
anonymous. Parse recovery and internal producer failures cannot create a
`ValidChatFile` capability.

Admission precedes payload collection and the dummy/no-payload pass-through
branches. Batch refusal keeps the producing failure category even if another
file's inference fails. The admitted document is stored with its payload slice;
editing consumes the immutable Chatter proof with `into_unchecked`.

The separate `coref` pipeline uses the same full retained-input admission before
the dummy and non-English pass-through branches. Each pending request owns its
admitted document and collected sentence mapping. Completion checks annotation
positions, per-sentence word counts and duplicate sentence ownership before
mutation. A `no_sentences` result for a non-empty dispatched document is a
protocol failure, not successful empty coreference. Resolved sparse output with
no chains remains legitimate.

Morphology selects its replacement plan from the headers of the same parse:
normal analysis discards `%mor`/`%gra` before retained admission, while honored
CA files retain and validate them. Incremental reuse fully admits its prior
document before copying analysis. No source substitution or second parse is
used to establish the removal exemption.

Comparison references carry an opaque `AdmittedComparisonReference`, created
only by complete source admission. The comparison recipe admits its reference
before calling the morphology gateway; benchmark admits its reference before
media preparation and ASR. Neither path reparses the reference downstream.

Alignment's unchanged-source branch requires complete source admission;
`NoAlign` is not an invalidity exemption. Active incremental alignment also
admits its prior once, before media preparation or UTR, and retains that
`RetainedFaPrior` across retries. An unreadable or invalid declared prior is a
refusal, not permission to silently switch to full alignment. A no-op branch
does not consume the optional prior.

Active alignment selects Chatter's adaptive word-timing plan from the headers
of the same source parse, before media preparation or UTR. Complete original
admission retains valid `%wor`, including partial timing. Otherwise concrete
source-bound word tiers are discarded, and every retained region must pass
normal structure and alignment checks. If recorded timing in those actual
discarded tiers was the only linkage evidence, a distinct pending timing state
carries the retained `@Media` obligation through regeneration. It does not
assert complete CHAT validity, filter diagnostics or modify headers. Final
output requires complete checked construction, including fulfilled linkage.
If that pending attempt produces no alignment requests because grouping refused
its acoustic window, it reports the typed window/budget refusal as unavailable
evidence before finalization. This is not a verdict that the submitted CHAT is
invalid. Chatter's shared, source-borrowing timing observer prevents this refusal
when UTR or checked prior-file reuse has already restored timing. Observation
is not validity. An unexplained missing-timing result still fails complete output
admission as an internal producer failure. Admitting a request plan does not
discharge the timing obligation or authorize writing.
CA retains all tiers;
NoAlign retains all tiers and its exact source bytes. Internal producer failures
never grant a regeneration exemption.

The owned working disposition produces active retry attempts or preserved-source
attempts. Only its active payload permits UTR mutation; no constructor pairs an
arbitrary recovered model with unrelated diagnostics or source bytes. Output
admission remains separate and complete after mutation. Media-error reporting
for align reads the already-produced headers rather than reparsing fragments.

## Completion and publication

Inference success is not successful application. Every required lexical item
must be covered by an admitted response and a completed transform before the
file can be published. Morphological injection returns a producer-issued
completion result or a typed refusal: empty ordinary lexical analysis, malformed
UD, count mismatch or failed retokenization cannot advance to applied analysis.
Explicit special-form synthesis and reported secondary-language placeholders
are separate, intentional outcomes; they are not silent missing morphology.

After mutation, validate the actual output structure and command guarantees.

For normal `align`, source admission owns the required lexical population.
Final output must retain correspondence with those source words and positive
timing for each, including after monotonicity and other finalization changes.
The producer's complete-result payload is required alongside CHAT admission;
restored media linkage or one timed word cannot substitute for completion.
Missing timing is an unavailable-evidence refusal, not invalid input or a
successful partial file. Valid declined-alignment paths retain their separate
unchanged-source capability. Timing coverage does not certify acoustic accuracy.

Do not retain a validity proof across edits. On failure of a command that
transforms submitted CHAT, write no partial CHAT file. Successful no-work input
retains its original bytes without inference or an invented processing stamp.

Output a generating producer built itself (transcription) is judged by the same
complete admission through `PostValidated::produced`, which never refuses: a
document that does not pass is a `DiagnosedOutput` (a different type from the
admitted `PostValidated`), written with every finding, and the file is reported
`diagnosed` (terminal, not retried, not a failure). Stages that require
admission take a `PostValidated` and so cannot receive it. An optional stage
whose own output is refused leaves the admitted pre-stage document, recorded as
a shortfall with a bounded record of the refusal; a file with any shortfall is
reported `diagnosed`. See
[CHAT validation failures](../developer/chat-validation-failures.md).

`PostValidated` is the write capability. Its transformed variants retain
Chatter's immutable checked-construction payload, with complete rules and tier
alignment. Applied analysis adds the command's completion checks; comparison
adds preservation checks. Neither preservation nor declined analysis can waive
CHAT validity, and callers cannot select an output `ValidityLevel`.

Unchanged output instead retains `AdmittedSourceChat`: original bytes and the
immutable model from that exact admitted parse. Independently supplied text
and model evidence cannot construct this capability. Editing consumes the
previous proof and requires fresh admission before writing. Test doubles use a
separate test-only state, not fabricated Chatter evidence.

Input invalidity, provider failure, incomplete model analysis and internal tool
failure are distinct. A failed model-to-CHAT transition is not evidence that the
submitted CHAT was invalid. Internal Chatter failures establish neither validity
nor invalidity. Preserve the producing category through per-file reporting and
retry decisions.

## Media-only commands

`transcribe` starts from audio rather than CHAT. It judges its assembled typed
transcript, and its final document, with complete admission through the
producer transition: an admitted transcript is published as done; one that is
not is published diagnosed, with every finding; segmentation and morphosyntax
still run on the utterances the findings do not belong to, and stages it cannot
run are skipped and recorded. Valid CHAT does not certify ASR accuracy; human review
remains necessary either way.
`benchmark` also has a reference-transcript contract in addition to ASR.
Audio-only feature commands have no CHAT-input exemption to apply.

Transcription carries its admitted model through optional post-CHAT segmentation,
morphology and benchmark handoff; a diagnosed model reaches segmentation and
morphology only outside the utterances its findings are confined to, and
benchmark, whose deliverable is a comparison that needs an admitted transcript,
refuses that file with the findings. These stages consume typed output rather than
serializing CHAT as their next stage's parser input. Diagnostic/debug dumps are
observations, not pipeline inputs. The final writer receives the output proof.

Media commands separately require admitted accessible media, supported engines,
measured duration where timing is produced and checked model/provider results.
These preconditions do not substitute for CHAT admission.

## Verification boundaries

Keep tests at the source-to-admitted-input, worker-to-completed-transform and
output-write boundaries. Required negative cases include invalid retraces,
retained corrupt generated tiers, invalid no-work/dummy inputs and failures in
one member of a mixed batch. Use inference doubles that fail if called for a
refused input; do not use timing or sleeps to infer that work was skipped.
Actual CLI/managed-server runs verify that outer dispatch paths preserve these
same guarantees.
