# Proportional FA Estimation: Removed

**Status:** Source repair; acoustic qualification remains separate
**Last updated:** 2026-10-15 18:33 EDT

FA grouping no longer chooses an audio window for an untimed
utterance by distributing words proportionally across the available audio.
This arithmetic does **not** establish that those words occur in that window.
A forced aligner can return positive, structurally valid intervals for supplied
text even when the selected audio contains different speech. Treating such
intervals as evidence that the window was correct is circular.

Earlier versions of this page claimed that an inaccurate window merely caused
slightly less accurate timing, and described the heuristic as already accurate
enough. Those claims are withdrawn. The historical coverage percentages also
did not establish acoustic accuracy and are removed.

## Current request boundary

`LocalizedUtterance` in `crates/batchalign/src/chat_ops/fa/grouping.rs` owns
the word population and its source-derived span. It requires a supplied
utterance bullet (including UTR-admitted timing), or a source-bound UTR
candidate envelope or partial-order search corridor checked against that exact
utterance/word population. Its constructor takes no fallback span.

Recording extent bounds the search; it is not used as a request window when
evidence is absent. Missing evidence produces `localization_unavailable`,
leaves the words untimed, and makes no backend request. Supported portions
remain processable, but grouping does not merge across an unsupported turn.
The localized population still needs recording and engine-budget admission,
including the existing anchored-splitting or explicit window-refusal rules.

Writable partial output reports `timing_incomplete` with the new cause.
A linked transcript with no retained timing cannot be written merely because
refusal was explicit: its timing obligation remains outstanding and the error
explains recovery or header remedies. Legacy `not_placed` records remain readable.

These checks establish different, limited facts:

| Check or evidence | What it establishes | What it does not establish |
|---|---|---|
| Recording bounds | The requested interval lies within the recording | The target speech is there |
| Supplied-timing density screen | The configured density limit did not reject the span | Natural speech rate or correct placement |
| Engine duration/label budget | The request meets those resource limits | Acoustic suitability |
| UTR lexical evidence | Correspondence under the selected matching model and retained ASR observations | Independent acoustic ground truth |
| Positive FA intervals | The model returned intervals that passed structural admission | The supplied words were spoken there |
| Serialized `%wor` and CHAT roundtrip | Timings survived output construction and the CHAT remains valid | Timing accuracy |

An original or UTR-supplied bullet also remains fallible. Removing proportional
estimation alone cannot guarantee that every remaining window or model result
is correct.
This repair does not retroactively validate or discard previously stored
timings. Existing word-timing reuse remains governed by its own admission
policy; suspect earlier outputs still need separate review.

## Why the fallback is unsafe

Word fractions assume that transcript order and cumulative word count track
recording time closely enough. Untranscribed introductions, silence, omitted
turns, overlap and variable speech rate can invalidate that assumption. A
two-second buffer supplies no general error bound. Splitting a long unknown
region into short guessed windows can make each request fit the engine budget
while losing the speech the request needs.

The original motivation was cheaper processing and fewer untimed words. Those
are useful only when the resulting timing is reliable. More positive intervals
cannot, by themselves, measure an improvement.

## Verification and its evidence boundary

The regression exercises the real FA entry point on a long synthetic untimed
transcript with worker startup deliberately unavailable. It checks that no
inference is attempted and the explicit cause reaches durable completion.
Existing supplied-window, UTR-search, partial-output and anchored-partition
tests cover supported routes. These are contract checks, not model-quality evidence.

Typestate binds the localized population to its words and source-derived
interval before request admission. A validating constructor can enforce that
policy; it cannot prove that an ASR hypothesis or model alignment is true.
Neither an `admitted` name nor a successful model response should erase this
distinction.

Output snapshots verify explicit refusal and serialization.
Acoustic evaluation additionally needs independently
checked word locations, with false placements and abstentions counted alongside
useful timings. Synthetic request tests alone do not provide that evaluation.

See [FA's current decision contract](../../batchalign/reference/forced-alignment.md#current-decision-contract-and-evidence-limits)
for the implemented decisions and remaining limits.
